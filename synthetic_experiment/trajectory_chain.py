"""Resumable task trajectory chain with BOLT and optional MI-ORPT milestones."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .prompts import messages
from .steps import (
    FINE_TUNING_DIR,
    _load_hf_model,
    _run,
    checkpoint_ready,
    cleanup_intermediate_epochs,
    materialize_hf_checkpoint,
    run_bo,
    sample_from_checkpoint,
    sampling_output_path,
    torchrun_master_port,
    tune_executable,
)
from .task_splits import task_name


def checkpoint_to_sample_from(cfg: ExperimentConfig, task_position: int) -> Path | None:
    completed = [m for m in cfg.milestones if m <= task_position]
    if not completed:
        return None
    latest = max(completed)
    return cfg.orpt_checkpoint_dir(latest) if cfg.build_orpt else cfg.milestone_checkpoint_dir(latest)


def build_sft_data(cfg: ExperimentConfig, milestone: int) -> Path:
    target = cfg.milestones_dir / f"train_data_{milestone}.jsonl"
    rows = []
    for index, task_t in enumerate(cfg.train_task_values[:milestone]):
        frame = pd.read_csv(cfg.trajectories_dir / f"{task_name(index)}.csv")
        ranked = frame.nlargest(cfg.sft_top_n_per_task, "train_y")
        for row in ranked.itertuples():
            point = json.loads(row.train_x)
            rows.append({"messages": messages(task_t, point)})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return target


def train_milestone(cfg: ExperimentConfig, milestone: int) -> Path:
    final = cfg.milestone_checkpoint_dir(milestone)
    if checkpoint_ready(final):
        return final
    data = build_sft_data(cfg, milestone)
    if cfg.proposal_source == "random":
        final.mkdir(parents=True, exist_ok=True)
        (final / "RANDOM_PROPOSAL_MARKER").write_text(f"sft_data={data}\n")
        return final
    from .orpt import check_training_size
    directory = cfg.bolt_root / FINE_TUNING_DIR
    check_training_size(data, directory / "torchtune_config" / cfg.torchtune_config, cfg.sft_epochs)
    out = cfg.checkpoints_dir / f"BOLT-{milestone}"
    _run([tune_executable(), "run", "--nnodes", "1", "--nproc_per_node", "1",
          "--master-port", str(torchrun_master_port(cfg)), cfg.torchtune_recipe,
          "--config", f"torchtune_config/{cfg.torchtune_config}", f"output_dir={out}",
          f"dataset.data_files={data}", f"checkpointer.checkpoint_dir={cfg.base_checkpoint_dir}",
          f"tokenizer.path={cfg.base_checkpoint_dir}/vocab.json",
          f"tokenizer.merges_file={cfg.base_checkpoint_dir}/merges.txt", f"epochs={cfg.sft_epochs}",
          "seed=42", f"metric_logger.log_dir={cfg.tensorboard_dir / f'BOLT-{milestone}'}"], directory, cfg)
    materialize_hf_checkpoint(out, cfg.sft_epochs - 1, cfg.base_checkpoint_dir)
    if not checkpoint_ready(final):
        raise RuntimeError(f"SFT did not produce {final}")
    cleanup_intermediate_epochs(out, cfg.sft_epochs - 1)
    return final


def _run_task_stage(cfg: ExperimentConfig, task_indices: list[int], checkpoint: Path | None) -> None:
    pending = [i for i in task_indices if not (cfg.trajectories_dir / f"{task_name(i)}.csv").exists()]
    if not pending:
        return

    initial_by_task = {}
    if checkpoint is not None and cfg.proposal_source == "llm":
        print(f"[chain] loading shared checkpoint once for {len(pending)} tasks: {checkpoint}", flush=True)
        loaded = _load_hf_model(checkpoint, cfg.base_checkpoint_dir)
        for index in pending:
            destination = cfg.trajectories_dir / f"{task_name(index)}.csv"
            initial_by_task[index] = sample_from_checkpoint(
                cfg,
                checkpoint,
                cfg.train_task_values[index],
                cfg.init_size,
                cfg.bo_seed + index,
                sampling_output_path(cfg, checkpoint, destination),
                loaded,
            )
        model = loaded[0]
        del loaded, model
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    workers = min(cfg.trajectory_parallel_workers, len(pending))
    print(f"[chain] running {len(pending)} task BO trajectories with {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="task-bo") as executor:
        futures = {}
        for index in pending:
            task_t = cfg.train_task_values[index]
            future = executor.submit(
                run_bo,
                cfg,
                task_t,
                cfg.trajectories_dir / f"{task_name(index)}.csv",
                seed=cfg.bo_seed + index,
                checkpoint=checkpoint,
                initial_x=initial_by_task.get(index),
            )
            futures[future] = index
        for future in as_completed(futures):
            index = futures[future]
            future.result()
            print(f"[chain] completed {task_name(index)}", flush=True)


def run_trajectory_chain(cfg: ExperimentConfig) -> None:
    cfg.ensure_dirs()
    start = 0
    boundaries = sorted(set(cfg.milestones + [len(cfg.train_task_values)]))
    for end in boundaries:
        if end <= start:
            continue
        checkpoint = checkpoint_to_sample_from(cfg, start)
        print(f"[chain] task interval [{start}, {end}) checkpoint={checkpoint}", flush=True)
        _run_task_stage(cfg, list(range(start, end)), checkpoint)
        if end in cfg.milestones:
            train_milestone(cfg, end)
            if cfg.build_orpt:
                from .orpt import train_orpt_milestone
                train_orpt_milestone(cfg, end)
        start = end
