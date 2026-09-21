"""The single shared BOLT-SFT trajectory chain: samples/BO's through
cfg.train_task_range(), training a fresh BOLT-<milestone> checkpoint each
time a configured milestone is reached. Every step is idempotent (skips work
whose output already exists), so an interrupted multi-day chain can just be
re-run to resume.

Parallel task dispatch (2026-09): tasks within one milestone segment (the
run between two consecutive milestones, or from 0 to the first one) all
sample from the SAME checkpoint and don't depend on each other's output, so
if cfg.mi_parallel_gpus is set, a segment's tasks run concurrently across
one dedicated worker thread per GPU (see _run_segment) instead of strictly
one task at a time -- the previous behavior, still used unmodified when
mi_parallel_gpus is unset. Each worker thread permanently owns one GPU for
the segment's duration (not a round-robin per-task assignment, which would
risk two tasks landing on the same physical GPU depending on how the
scheduler happens to interleave variable-duration tasks).

Warm sampling pool (2026-09): LLM sampling is no longer a subprocess per
task. _run_segment stands up one persistent worker per GPU holding that
segment's single sampling checkpoint resident (mi_orpt/warm_sampling_pool.py)
and tears it down in a finally before run_trajectory_chain() reaches
train_milestone -- a `tune run` needs 27-36GB/GPU and will not fit alongside
the pool's ~6GB. Measured motivation: sampling was ~38s/call for ~7s of
generation, the rest being repeated checkpoint loads.

Known partial limitation: steps.py::sample_and_build_init/build_mutation_init
dispatch their remaining expensive work (BO itself) via subprocess, which
correctly picks up each worker thread's own pinned cuda_visible_devices
(steps.py::_run passes a per-call env dict, no shared mutable state between
concurrent calls). Their small in-process oracle
top-up call (steps.py::_ensure_constraint_feasible, and build_mutation_init
itself for tasks before the first milestone) calls the APEX oracle directly
in the *driver* process, not a subprocess -- CUDA_VISIBLE_DEVICES is read
once when a process's CUDA context first initializes, so that one piece
silently stays on whichever GPU happened to bind first, for every worker,
regardless of which one is nominally running. This is a real inefficiency
(no 4-way speedup for that one small piece) but not a correctness bug --
it's a bounded number of short inference-only calls (scoring at most
cfg.init_size candidates), a small fraction of a task's total wall time
next to the dominant real-BO subprocess cost.
"""

from __future__ import annotations

import dataclasses
import queue
import sys
import threading
from pathlib import Path

from .config import ExperimentConfig
from .mi_orpt.warm_sampling_pool import create_sampling_pools, shutdown_sampling_pools
from .orpt import train_orpt_milestone
from .steps import (
    _run,
    build_mutation_init,
    cleanup_intermediate_epochs,
    distributed_finetune_launch,
    materialize_hf_checkpoint,
    sample_and_build_init,
    run_bo,
)

FINE_TUNING_DIR = "fine-tuning/peptides"


def checkpoint_to_sample_from(cfg: ExperimentConfig, task_idx: int) -> Path | None:
    """The milestone checkpoint to sample task `task_idx` from: the most
    recently completed milestone strictly before this task, or None if no
    milestone has been reached yet (caller should use build_mutation_init
    instead of sampling an untuned model -- see build_mutation_init's
    docstring).

    When cfg.build_orpt is True, this returns that milestone's ORPT-<m>
    checkpoint instead of BOLT-<m> -- ORPT's DPO output becomes the ongoing
    "current policy" that samples subsequent tasks (confirmed with the user
    against the original peptide_script_dpo.sh's self-reinforcing loop
    design; BOLT-<m> itself is still built at every milestone but is only
    ever used as ORPT's init/ref, never for sampling, once ORPT is enabled).
    """
    done_milestones = [m for m in cfg.milestones if m <= task_idx]
    if not done_milestones:
        return None
    latest = max(done_milestones)
    if cfg.build_orpt:
        return cfg.orpt_checkpoint_dir(latest)
    return cfg.milestone_checkpoint_dir(latest)


def train_milestone(cfg: ExperimentConfig, milestone: int) -> Path:
    """Build the cumulative SFT dataset from tasks [0, milestone) and train
    a BOLT-<milestone> checkpoint from the base model.
    """
    ckpt_dir = cfg.checkpoints_dir / f"BOLT-{milestone}"
    final_ckpt = cfg.milestone_checkpoint_dir(milestone)
    if (final_ckpt / "model.safetensors.index.json").exists():
        print(f"[milestone {milestone}] checkpoint already exists at {final_ckpt}, skipping SFT")
        return final_ckpt

    fine_tuning_dir = cfg.bolt_root / FINE_TUNING_DIR
    train_csv = cfg.milestones_dir / f"train_data_{milestone}.csv"
    train_jsonl = cfg.milestones_dir / f"train_data_{milestone}.jsonl"

    input_csvs = [cfg.trajectories_csv_dir / f"task_{i:04d}.csv" for i in range(milestone)]
    reference_indices = [str(i) for i in range(milestone)]

    _run(
        [
            sys.executable,
            "make_train_data_csv.py",
            "--input-csv",
            *input_csvs,
            "--reference-index",
            *reference_indices,
            "--output-csv",
            train_csv,
            "--similarity-threshold",
            cfg.similarity_threshold,
        ],
        cwd=fine_tuning_dir,
        cfg=cfg,
    )
    _run(
        [
            sys.executable,
            "generate_openai_ft_data.py",
            "--data-path",
            train_csv,
            "--save-path",
            train_jsonl,
        ],
        cwd=fine_tuning_dir,
        cfg=cfg,
    )
    torchrun_flags, extra_overrides, launch_cfg = distributed_finetune_launch(cfg, cfg.torchtune_config, fine_tuning_dir)
    nproc_per_node = len(cfg.mi_parallel_gpus) if cfg.mi_parallel_gpus else 1
    _run(
        [
            "tune",
            "run",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(nproc_per_node),
            *torchrun_flags,
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
            *extra_overrides,
        ],
        cwd=fine_tuning_dir,
        cfg=launch_cfg,
    )
    materialize_hf_checkpoint(ckpt_dir, cfg.sft_epochs - 1, cfg.base_checkpoint_dir)
    if not (final_ckpt / "model.safetensors.index.json").exists():
        raise RuntimeError(
            f"Expected milestone {milestone} checkpoint at {final_ckpt}, but it wasn't produced"
        )
    cleanup_intermediate_epochs(ckpt_dir, cfg.sft_epochs - 1)
    return final_ckpt


def _run_single_task(cfg: ExperimentConfig, task_idx: int, *, sampling_pool=None) -> None:
    """One task's own init-building + full BO trajectory. cfg is expected to
    already carry whichever cuda_visible_devices this call should pin to
    (see _run_segment). sampling_pool, when given, is this GPU's warm sampling
    worker -- irrelevant to the build_mutation_init branch, which never touches
    the LLM."""
    model_path = checkpoint_to_sample_from(cfg, task_idx)
    if model_path is None:
        init_path, scores_path = build_mutation_init(cfg, task_idx, cfg.trajectories_dir)
    else:
        init_path, scores_path = sample_and_build_init(
            cfg, model_path, task_idx, cfg.trajectories_dir, sampling_pool=sampling_pool
        )
    run_bo(
        cfg,
        task_idx,
        cfg.trajectories_csv_dir,
        run_id="chain",
        init_path=init_path,
        scores_path=scores_path,
    )


def _run_segment(cfg: ExperimentConfig, task_indices: list[int]) -> None:
    """Runs every task in task_indices (all sampling from the same
    checkpoint -- see checkpoint_to_sample_from), in parallel across
    cfg.mi_parallel_gpus if set: one dedicated worker thread per GPU, each
    permanently pinned to that GPU for the segment's duration, pulling
    tasks off a shared queue until it's empty (module docstring explains
    why -- not a round-robin per-task assignment). Falls back to plain
    serial dispatch (the original, still-default behavior) when
    mi_parallel_gpus is unset.

    On any task's failure, already-dispatched tasks are allowed to finish
    (mirrors ThreadPoolExecutor's own shutdown(wait=True) semantics -- no
    new tasks start once a failure is seen, but in-flight subprocesses
    aren't killed mid-run) before the first failure is raised.
    """
    if not task_indices:
        return
    gpus = cfg.mi_parallel_gpus
    if not gpus:
        for task_idx in task_indices:
            _run_single_task(cfg, task_idx)
        return

    # Every task in a segment samples from the same checkpoint, which is what
    # makes one resident model per GPU valid for the whole segment. Assert it
    # rather than assume it: a pool built for the wrong checkpoint would emit
    # plausible-looking samples from a stale model with no other symptom.
    checkpoints = {checkpoint_to_sample_from(cfg, task_idx) for task_idx in task_indices}
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"_run_segment: tasks {task_indices[0]}..{task_indices[-1]} span multiple "
            f"sampling checkpoints {checkpoints} -- segments must not cross a milestone"
        )
    model_path = checkpoints.pop()

    pools: dict = {}
    if model_path is not None:
        # model_path is None for every task before the first milestone: that
        # segment is entirely build_mutation_init, which uses no LLM, so a pool
        # would load a checkpoint that need not even exist yet.
        if not (model_path / "model.safetensors.index.json").exists():
            raise FileNotFoundError(
                f"_run_segment: {model_path} has no model.safetensors.index.json; refusing to "
                "start a warm sampling pool on it (ProcessPoolExecutor runs its initializer "
                "lazily, so this would otherwise surface as an opaque BrokenProcessPool)"
            )
        pools = create_sampling_pools(gpus, cfg.bolt_root / FINE_TUNING_DIR, model_path)

    work_queue: queue.Queue[int] = queue.Queue()
    for task_idx in task_indices:
        work_queue.put(task_idx)
    errors: list[tuple[int, BaseException]] = []
    errors_lock = threading.Lock()
    stop_event = threading.Event()

    def worker(gpu: str) -> None:
        task_cfg = dataclasses.replace(cfg, cuda_visible_devices=gpu)
        sampling_pool = pools.get(gpu)
        while not stop_event.is_set():
            try:
                task_idx = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                _run_single_task(task_cfg, task_idx, sampling_pool=sampling_pool)
            except Exception as e:  # noqa: BLE001 -- collected, re-raised below, not swallowed
                with errors_lock:
                    errors.append((task_idx, e))
                stop_event.set()
                return

    try:
        threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            task_idx, exc = errors[0]
            raise RuntimeError(f"trajectory chain task {task_idx} failed: {exc!r}") from exc
    finally:
        # MUST happen before run_trajectory_chain() moves on to train_milestone/
        # train_orpt_milestone: a `tune run` peaks near 27GB/GPU (SFT) or 36GB
        # (DPO), and 36 + this pool's ~6GB resident checkpoint does not fit on a
        # 40GB card. finally (not `with`) because there are several pools and
        # because the error path above re-raises.
        shutdown_sampling_pools(pools)


def run_trajectory_chain(cfg: ExperimentConfig) -> None:
    cfg.ensure_dirs()
    segment: list[int] = []
    for task_idx in cfg.train_task_range():
        segment.append(task_idx)
        completed_count = task_idx + 1
        if completed_count in cfg.milestones:
            _run_segment(cfg, segment)
            segment = []
            train_milestone(cfg, completed_count)
            if cfg.build_orpt:
                # Must happen before the next task is sampled: once ORPT is
                # enabled, checkpoint_to_sample_from() needs ORPT-<completed_count>
                # to already exist.
                train_orpt_milestone(cfg, completed_count)
    _run_segment(cfg, segment)
