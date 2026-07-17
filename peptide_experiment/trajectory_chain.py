"""The single shared BOLT-SFT trajectory chain: samples/BO's through
cfg.train_task_range(), training a fresh BOLT-<milestone> checkpoint each
time a configured milestone is reached. Every step is idempotent (skips work
whose output already exists), so an interrupted multi-day chain can just be
re-run to resume.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import ExperimentConfig
from .orpt import train_orpt_milestone
from .steps import _run, build_mutation_init, cleanup_intermediate_epochs, sample_and_build_init, run_bo

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
    if final_ckpt.exists():
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
        raise RuntimeError(
            f"Expected milestone {milestone} checkpoint at {final_ckpt}, but it wasn't produced"
        )
    cleanup_intermediate_epochs(ckpt_dir, final_ckpt)
    return final_ckpt


def run_trajectory_chain(cfg: ExperimentConfig) -> None:
    cfg.ensure_dirs()
    for task_idx in cfg.train_task_range():
        model_path = checkpoint_to_sample_from(cfg, task_idx)
        if model_path is None:
            init_path, scores_path = build_mutation_init(
                cfg, task_idx, cfg.trajectories_dir
            )
        else:
            init_path, scores_path = sample_and_build_init(
                cfg, model_path, task_idx, cfg.trajectories_dir
            )
        run_bo(
            cfg,
            task_idx,
            cfg.trajectories_csv_dir,
            run_id="chain",
            init_path=init_path,
            scores_path=scores_path,
        )

        completed_count = task_idx + 1
        if completed_count in cfg.milestones:
            train_milestone(cfg, completed_count)
            if cfg.build_orpt:
                # Must happen before the next task is sampled: once ORPT is
                # enabled, checkpoint_to_sample_from() needs ORPT-<completed_count>
                # to already exist.
                train_orpt_milestone(cfg, completed_count)
