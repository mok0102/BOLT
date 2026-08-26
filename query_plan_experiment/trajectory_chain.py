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
from .steps import _run, cleanup_intermediate_epochs, run_bo, sample_and_build_init

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
    return cfg.milestone_checkpoint_dir(max(done_milestones))


def train_milestone(cfg: ExperimentConfig, milestone: int) -> Path:
    """Build the cumulative SFT dataset from workloads [0, milestone) and
    train a BOLT-<milestone> checkpoint from the base model."""
    ckpt_dir = cfg.checkpoints_dir / f"BOLT-{milestone}"
    final_ckpt = cfg.milestone_checkpoint_dir(milestone)
    if final_ckpt.exists():
        print(f"[milestone {milestone}] checkpoint already exists at {final_ckpt}, skipping SFT")
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
        ],
        cwd=fine_tuning_dir,
        cfg=cfg,
    )
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
            run_bo(cfg, workload, cfg.trajectories_csv_dir, run_id="chain", init_csv_path=init_csv_path)

        completed_count = i + 1
        if completed_count in cfg.milestones:
            train_milestone(cfg, completed_count)
