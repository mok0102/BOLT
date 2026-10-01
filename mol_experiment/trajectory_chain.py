"""The single shared BOLT-SFT trajectory chain for the mol domain: samples/
BO's through cfg.train_task_range(), training a fresh BOLT-<milestone>
checkpoint each time a configured milestone is reached. Mirrors
peptide_experiment/trajectory_chain.py's shape (read-only reference, never
imported -- isolation contract): every step is idempotent (skips work whose
output already exists), so an interrupted multi-milestone chain can just be
re-run to resume.

_run_single_task branches exactly like peptide's own version:
checkpoint_to_sample_from's None branch (no milestone reached yet) uses
mol_build_mutation_init; its non-None branch uses mol_sample_and_build_init
(mol_experiment/steps.py), sampling from that milestone's own BOLT-<m> (or
ORPT-<m>, once cfg.build_orpt is enabled) checkpoint.

_run_segment parallelizes _run_single_task across cfg.mi_parallel_gpus (one
worker process per GPU, mol_experiment/warm_task_pool.py -- see that
module's own docstring for the real profiling that motivated this: at
production scale, mol_run_bo's own per-task cost (~8 min, dominated by
candidate generation, not GP fitting) was paid ENTIRELY serially on one GPU
while the rest of mi_parallel_gpus sat idle for the whole pre-milestone
phase). Every task within a segment is independent (same sampling
checkpoint or none), so this is a correctness-preserving, real wall-clock
win -- not an approximation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import MolExperimentConfig
from .orpt import train_orpt_milestone
from .steps import (
    _run,
    cleanup_intermediate_epochs,
    distributed_finetune_launch,
    materialize_hf_checkpoint,
    mol_build_mutation_init,
    mol_run_bo,
    mol_sample_and_build_init,
)
from .warm_task_pool import _run_task_in_worker, create_task_pool

FINE_TUNING_DIR = "fine-tuning/mol"


def checkpoint_to_sample_from(cfg: MolExperimentConfig, task_idx: int) -> Path | None:
    """The milestone checkpoint to sample task `task_idx` from: the most
    recently completed milestone strictly before this task, or None if no
    milestone has been reached yet (_run_single_task then uses
    mol_build_mutation_init instead of sampling an untuned model).

    When cfg.build_orpt is True, this returns that milestone's ORPT-<m>
    checkpoint instead of BOLT-<m> (mirrors peptide's own self-reinforcing-
    loop design: ORPT's DPO output becomes the ongoing "current policy"
    that samples subsequent tasks once ORPT is enabled).
    """
    done_milestones = [m for m in cfg.milestones if m <= task_idx]
    if not done_milestones:
        return None
    latest = max(done_milestones)
    if cfg.build_orpt:
        return cfg.orpt_checkpoint_dir(latest)
    return cfg.milestone_checkpoint_dir(latest)


def train_milestone(cfg: MolExperimentConfig, milestone: int) -> Path:
    """Build the cumulative SFT dataset from tasks [0, milestone) and train
    a BOLT-<milestone> checkpoint from the base model."""
    ckpt_dir = cfg.checkpoints_dir / f"BOLT-{milestone}"
    final_ckpt = cfg.milestone_checkpoint_dir(milestone)
    if (final_ckpt / "model.safetensors.index.json").exists():
        print(f"[milestone {milestone}] checkpoint already exists at {final_ckpt}, skipping SFT")
        return final_ckpt

    fine_tuning_dir = cfg.bolt_root / FINE_TUNING_DIR
    train_csv = cfg.milestones_dir / f"train_data_{milestone}.csv"
    train_jsonl = cfg.milestones_dir / f"train_data_{milestone}.jsonl"

    input_csvs = [cfg.trajectories_csv_dir / f"task_{i:04d}.csv" for i in range(milestone)]
    task_indices = [str(i) for i in range(milestone)]

    _run(
        [
            sys.executable,
            "make_train_data_csv.py",
            "--input-csv",
            *input_csvs,
            "--task-index",
            *task_indices,
            "--output-csv",
            train_csv,
            "--tau-mol",
            cfg.tau_mol,
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
        raise RuntimeError(f"Expected milestone {milestone} checkpoint at {final_ckpt}, but it wasn't produced")
    cleanup_intermediate_epochs(ckpt_dir, cfg.sft_epochs - 1)
    return final_ckpt


def _run_single_task(cfg: MolExperimentConfig, task_idx: int) -> None:
    """One task's own init-building + full BO trajectory. Uses
    mol_build_mutation_init before any milestone is reached, and
    mol_sample_and_build_init (real LLM sampling from the most recent
    milestone checkpoint) afterward -- mirrors peptide's own
    _run_single_task exactly."""
    model_path = checkpoint_to_sample_from(cfg, task_idx)
    if model_path is None:
        init_path, scores_path = mol_build_mutation_init(cfg, task_idx, cfg.trajectories_dir)
    else:
        init_path, scores_path = mol_sample_and_build_init(cfg, model_path, task_idx, cfg.trajectories_dir)
    mol_run_bo(
        cfg,
        task_idx,
        cfg.trajectories_csv_dir,
        run_id="chain",
        init_path=init_path,
        scores_path=scores_path,
    )


def _run_segment(cfg: MolExperimentConfig, task_indices: list[int]) -> None:
    """Runs every task in task_indices (all sampling from the same
    checkpoint -- see checkpoint_to_sample_from) in parallel across
    cfg.mi_parallel_gpus if set: one dedicated worker process per GPU
    (mol_experiment/warm_task_pool.py), pulling tasks off the pool until
    every task in task_indices is done. Falls back to plain serial dispatch
    (mi_parallel_gpus unset) otherwise -- the original, still-supported
    behavior.

    On any task's failure, already-dispatched tasks are allowed to finish
    (ProcessPoolExecutor has no "cancel in-flight work" primitive) before
    the first failure is raised.
    """
    if not task_indices:
        return
    gpus = cfg.mi_parallel_gpus
    if not gpus:
        for task_idx in task_indices:
            _run_single_task(cfg, task_idx)
        return

    # Every task in a segment samples from the same checkpoint, which is what
    # makes independent concurrent dispatch valid. Assert it rather than
    # assume it -- a segment spanning a milestone boundary would silently mix
    # two different "current policy" checkpoints with no other symptom.
    checkpoints = {checkpoint_to_sample_from(cfg, task_idx) for task_idx in task_indices}
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"_run_segment: tasks {task_indices[0]}..{task_indices[-1]} span multiple "
            f"sampling checkpoints {checkpoints} -- segments must not cross a milestone"
        )

    pool = create_task_pool(gpus)
    try:
        futures = {pool.submit(_run_task_in_worker, {"cfg": cfg, "task_idx": i}): i for i in task_indices}
        errors: list[tuple[int, str]] = []
        for future in futures:
            ok, err = future.result()
            if not ok:
                errors.append((futures[future], err))
        if errors:
            task_idx, err = errors[0]
            raise RuntimeError(f"trajectory chain task {task_idx} failed: {err}")
    finally:
        pool.shutdown(wait=True)


def run_trajectory_chain(cfg: MolExperimentConfig) -> None:
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
                # Must happen before the next segment is sampled once LLM
                # sampling is used: checkpoint_to_sample_from() needs
                # ORPT-<completed_count> to already exist by then.
                train_orpt_milestone(cfg, completed_count)
    _run_segment(cfg, segment)
