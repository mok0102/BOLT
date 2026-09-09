"""The single shared BOLT-SFT trajectory chain over the 1426-task training
workload, training a fresh BOLT-<milestone> checkpoint each time a
configured milestone is reached. Mirrors
peptide_experiment/trajectory_chain.py: every step is idempotent (skips work
whose output already exists), so an interrupted multi-day chain can just be
re-run to resume. No ORPT branch -- out of scope for this pass, see
imp_plan/02_query_plan_reimplementation_plan.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import ExperimentConfig
from .timing import timed_operation
from query_plan_timing import event
from .steps import WORKLOAD_DIR, _run, cleanup_intermediate_epochs, run_bo, sample_and_build_init

FINE_TUNING_DIR = "fine-tuning/query_plans"


def checkpoint_to_sample_from(cfg: ExperimentConfig, workload_position: int) -> Path | None:
    """The milestone checkpoint to sample the workload at `workload_position`
    from: the most recently completed milestone strictly before this
    position, or None if no milestone has been reached yet (caller should
    use the LOLBO script's own init_w_bao=True default instead of sampling
    an untuned model -- see run_trajectory_chain)."""
    done_milestones = [m for m in cfg.milestones if m <= workload_position]
    if not done_milestones:
        return None
    latest = max(done_milestones)
    return cfg.orpt_checkpoint_dir(latest) if cfg.build_orpt else cfg.milestone_checkpoint_dir(latest)


@timed_operation("sft")
def train_milestone(cfg: ExperimentConfig, milestone: int) -> Path:
    """Build the cumulative SFT dataset from workloads [0, milestone) and
    train a BOLT-<milestone> checkpoint from the base model."""
    ckpt_dir = cfg.checkpoints_dir / f"BOLT-{milestone}"
    final_ckpt = cfg.milestone_checkpoint_dir(milestone)
    if final_ckpt.exists():
        print(f"[milestone {milestone}] checkpoint already exists at {final_ckpt}, skipping SFT")
        event("cache_hit", 0, artifact=str(final_ckpt))
        return final_ckpt

    fine_tuning_dir = cfg.bolt_root / FINE_TUNING_DIR
    workloads = cfg.train_task_workloads()[:milestone]
    train_csv = cfg.milestones_dir / f"train_data_{milestone}.csv"
    train_jsonl = cfg.milestones_dir / f"train_data_{milestone}.jsonl"

    input_csvs = [cfg.trajectories_csv_dir / f"{w}.csv" for w in workloads]

    # SFT data-prep (make_train_data_csv.py/generate_openai_ft_data.py) and
    # torchtune both run in the main environment, not the LOLBO one -- unlike
    # steps.py's run_bo()/sample_and_build_init(), never cfg.lolbo_python.
    python = sys.executable
    _run(
        [
            python,
            "make_train_data_csv.py",
            "--input-csv",
            *input_csvs,
            "--workload-name",
            *workloads,
            "--output-csv",
            train_csv,
            "--top-n",
            cfg.sft_top_n_per_task,
        ],
        cwd=fine_tuning_dir,
        cfg=cfg,
    )
    _run(
        [
            python,
            "generate_openai_ft_data.py",
            "--data-path",
            train_csv,
            "--save-path",
            train_jsonl,
            "--workload-dir",
            cfg.bolt_root / WORKLOAD_DIR,
        ],
        cwd=fine_tuning_dir,
        cfg=cfg,
    )
    from .orpt import check_training_size
    check_training_size(train_jsonl, fine_tuning_dir / "torchtune_config" / cfg.torchtune_config, cfg.sft_epochs)
    _run(
        [
            "tune",
            "run",
            "--nnodes",
            "1",
            "--nproc_per_node",
            "1",
            cfg.torchtune_recipe,
            "--config",
            f"torchtune_config/{cfg.torchtune_config}",
            f"output_dir={ckpt_dir}",
            f"dataset.data_files={train_jsonl}",
            f"tokenizer.path={cfg.base_checkpoint_dir}/vocab.json",
            f"tokenizer.merges_file={cfg.base_checkpoint_dir}/merges.txt",
            f"checkpointer.checkpoint_dir={cfg.base_checkpoint_dir}",
            f"epochs={cfg.sft_epochs}",
            "seed=42",
            f"metric_logger.log_dir={cfg.tensorboard_dir / f'BOLT-{milestone}'}",
        ],
        cwd=fine_tuning_dir,
        cfg=cfg,
    )
    if not final_ckpt.exists():
        raise RuntimeError(f"Expected milestone {milestone} checkpoint at {final_ckpt}, but it wasn't produced")
    cleanup_intermediate_epochs(ckpt_dir, final_ckpt)
    return final_ckpt


def run_trajectory_chain(cfg: ExperimentConfig) -> None:
    cfg.ensure_dirs()
    workloads = cfg.train_task_workloads()
    for i, workload in enumerate(workloads):
        model_path = checkpoint_to_sample_from(cfg, i)
        if model_path is None:
            # Pre-first-milestone: no fine-tuned checkpoint exists yet. The
            # LOLBO script's own init_w_bao=True default already reads real,
            # already-scored BAO plans per-workload -- no init_csv_path
            # override needed at all (unlike peptide, which needs a
            # dedicated mutation-based fallback function here).
            run_bo(cfg, workload, cfg.trajectories_csv_dir, run_id="chain")
        else:
            init_csv_path = sample_and_build_init(cfg, model_path, workload, cfg.trajectories_dir)
            run_bo(cfg, workload, cfg.trajectories_csv_dir, run_id="chain", init_csv_path=init_csv_path, init_w_llm=True)

        completed_count = i + 1
        if completed_count in cfg.milestones:
            train_milestone(cfg, completed_count)
            if cfg.build_orpt:
                from .orpt import train_orpt_milestone
                train_orpt_milestone(cfg, completed_count)


def _run_chain_batch_lane(cfg, jobs, model_path, gpu_id):
    """One spawned worker with a fixed GPU; tasks in a lane run sequentially."""
    import os
    from dataclasses import replace

    if gpu_id is not None:
        cfg = replace(cfg, cuda_visible_devices=str(gpu_id))
    if cfg.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    completed = []
    for workload in jobs:
        destination = cfg.trajectories_csv_dir / f"{workload}.csv"
        if destination.exists():
            print(f"[batch {workload}] existing trajectory, skipping", flush=True)
        elif model_path is None:
            run_bo(cfg, workload, cfg.trajectories_csv_dir, run_id="chain")
        else:
            init_path = sample_and_build_init(cfg, model_path, workload, cfg.trajectories_dir)
            run_bo(cfg, workload, cfg.trajectories_csv_dir, run_id="chain",
                   init_csv_path=init_path, init_w_llm=True)
        if not destination.is_file():
            raise RuntimeError(f"Missing completed trajectory: {destination}")
        completed.append(workload)
    return completed


def _run_chain_batch_stage(cfg, jobs, model_path, workers, gpu_ids):
    """Spawn isolated processes; wait for every lane before returning."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    pending = [w for w in jobs if not (cfg.trajectories_csv_dir / f"{w}.csv").is_file()]
    if not pending:
        return
    n_workers = min(workers, len(pending))
    # A fixed lane avoids moving a CUDA-initialized worker between GPUs.
    lanes = [pending[i::n_workers] for i in range(n_workers)]
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn"),
                             max_tasks_per_child=1) as pool:
        futures = {
            pool.submit(_run_chain_batch_lane, cfg, lane, model_path,
                        gpu_ids[i] if gpu_ids else None): lane
            for i, lane in enumerate(lanes)
        }
        for future in as_completed(futures):
            try:
                finished = future.result()
            except Exception as exc:
                for other in futures:
                    other.cancel()
                raise RuntimeError(f"Batch lane failed for {futures[future]}; milestone training not started") from exc
            print(f"[batch] completed {len(finished)} workloads: {finished}", flush=True)


def _run_trajectory_chain_batch(cfg: ExperimentConfig, workers: int = 2, gpu_ids=None) -> None:
    """Parallel tasks within each milestone interval, followed by serial SFT/ORPT.

    gpu_ids is an optional list or comma-separated CUDA_VISIBLE_DEVICES IDs.
    With no IDs, each worker inherits cfg/the shell's device visibility.
    The public batch entry point serializes DB evaluation with a shared gate.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if isinstance(gpu_ids, str):
        gpu_ids = [x.strip() for x in gpu_ids.split(",") if x.strip()]
    elif gpu_ids is not None:
        gpu_ids = [str(x) for x in gpu_ids]
    if gpu_ids and (len(set(gpu_ids)) != len(gpu_ids) or workers > len(gpu_ids)):
        raise ValueError("Provide distinct GPU IDs, at least one per worker")
    cfg.ensure_dirs()
    workloads = cfg.train_task_workloads()
    milestones = sorted(set(cfg.milestones))
    if any(m < 1 for m in milestones):
        raise ValueError("milestones must be positive task counts")
    boundaries = sorted({m for m in milestones if m <= len(workloads)} | {len(workloads)})
    start = 0
    for end in boundaries:
        if start == end:
            continue
        model_path = checkpoint_to_sample_from(cfg, start)
        if model_path is not None and not model_path.is_dir():
            raise RuntimeError(f"Missing previous milestone checkpoint: {model_path}")
        print(f"[batch] tasks {start+1}..{end}, workers={workers}, checkpoint={model_path}", flush=True)
        _run_chain_batch_stage(cfg, workloads[start:end], model_path, workers, gpu_ids)
        missing = [w for w in workloads[start:end] if not (cfg.trajectories_csv_dir / f"{w}.csv").is_file()]
        if missing:
            raise RuntimeError(f"Incomplete milestone interval: {missing}")
        if end in milestones:
            train_milestone(cfg, end)
            if cfg.build_orpt:
                from .orpt import train_orpt_milestone
                train_orpt_milestone(cfg, end)
        start = end


def run_trajectory_chain_batch(cfg: ExperimentConfig, workers: int = 2, gpu_ids=None) -> None:
    """Parallel model work, with one DB evaluation at a time across workers.

    Scope the gate to this entry point and its child processes. The ordinary
    run_trajectory_chain does not enable it. Waiting occurs outside SQL timing.
    """
    import hashlib
    import os
    import tempfile
    from oracle.batch_gate import ENV_NAME

    previous = os.environ.get(ENV_NAME)
    # The same local DB target shares a gate across batch experiment IDs.
    # No credentials are stored in the filename.
    target = "|".join(os.environ.get(k, "") for k in ("DB_HOST", "PGHOST", "PGPORT", "PGDATABASE"))
    key = hashlib.sha256(target.encode()).hexdigest()[:16]
    path = previous or str(Path(tempfile.gettempdir()) / f"bolt-query-db-{key}.lock")
    os.environ[ENV_NAME] = path
    try:
        print(f"[batch] DB evaluations serialized via {path}; model work remains parallel", flush=True)
        return _run_trajectory_chain_batch(cfg, workers, gpu_ids)
    finally:
        if previous is None:
            os.environ.pop(ENV_NAME, None)
        else:
            os.environ[ENV_NAME] = previous
