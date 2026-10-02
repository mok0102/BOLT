"""Build matched-intervention pairs and train ORPT at a milestone."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import yaml

from .config import ExperimentConfig
from .mi_orpt.candidate_bank import build_eligible_bank
from .mi_orpt.likelihood import score_sequences
from .mi_orpt.pair_construction import construct_pairs_for_task
from .prompts import messages
from .steps import (
    FINE_TUNING_DIR, _run, checkpoint_ready, cleanup_intermediate_epochs,
    materialize_hf_checkpoint, torchrun_master_port, tune_executable,
)


def check_training_size(data: Path, recipe: Path, epochs: int) -> None:
    rows = sum(bool(line.strip()) for line in data.read_text().splitlines())
    settings = yaml.safe_load(recipe.read_text())
    batch = int(settings["batch_size"])
    accumulation = int(settings.get("gradient_accumulation_steps", 1))
    if epochs < 1 or (rows // batch) // accumulation < 1:
        raise RuntimeError(f"No optimizer steps: {rows} rows, batch={batch}, accumulation={accumulation}")


def build_orpt_pairs(cfg: ExperimentConfig, milestone: int) -> Path:
    target = cfg.orpt_pairs_dir / f"orpt_pairs_{milestone}.jsonl"
    if target.exists() and target.stat().st_size:
        print(f"[ORPT pairs m={milestone}] cached: {target}", flush=True)
        return target
    records = []
    tasks = cfg.train_tasks[:milestone]
    total_tasks = len(tasks)
    milestone_started = time.perf_counter()
    print(
        f"[ORPT pairs m={milestone}] start: tasks={total_tasks}, "
        f"candidates/task={cfg.mi_max_candidates_per_task}, "
        f"backgrounds={cfg.mi_num_backgrounds}, target_pairs/task={cfg.mi_target_pairs_per_task}",
        flush=True,
    )
    for index, task in enumerate(tasks):
        position = index + 1
        name = task.name
        task_started = time.perf_counter()
        bank = build_eligible_bank(cfg.trajectories_dir / f"{name}.csv")
        print(
            f"[ORPT pairs m={milestone}] task {position}/{total_tasks} "
            f"{name} bank={len(bank)}: scoring likelihoods",
            flush=True,
        )
        if len(bank) < 3:
            elapsed = time.perf_counter() - task_started
            print(
                f"[ORPT pairs m={milestone}] task {position}/{total_tasks} {name}: "
                f"skipped (bank < 3), elapsed={elapsed:.1f}s",
                flush=True,
            )
            continue
        scoring_started = time.perf_counter()
        likelihoods = score_sequences(
            cfg, cfg.milestone_checkpoint_dir(milestone), task.transform, [c.seq for c in bank],
        )
        scoring_seconds = time.perf_counter() - scoring_started
        print(
            f"[ORPT pairs m={milestone}] task {position}/{total_tasks} {name}: "
            f"likelihoods done in {scoring_seconds:.1f}s; running MI evaluations",
            flush=True,
        )
        mi_started = time.perf_counter()
        task_records = construct_pairs_for_task(
            cfg, task, bank, likelihoods, random.Random(cfg.mi_seed + index),
            cfg.run_dir / "mi_evaluations" / f"milestone_{milestone}" / name,
        )
        mi_seconds = time.perf_counter() - mi_started
        records.extend(task_records)
        task_seconds = time.perf_counter() - task_started
        milestone_elapsed = time.perf_counter() - milestone_started
        average_seconds = milestone_elapsed / position
        eta_seconds = average_seconds * (total_tasks - position)
        print(
            f"[ORPT pairs m={milestone}] task {position}/{total_tasks} {name}: "
            f"pairs={len(task_records)}/{cfg.mi_target_pairs_per_task}, "
            f"scoring={scoring_seconds:.1f}s, mi={mi_seconds:.1f}s, total={task_seconds:.1f}s, "
            f"cumulative_pairs={len(records)}, elapsed={milestone_elapsed / 60:.1f}m, "
            f"eta={eta_seconds / 60:.1f}m",
            flush=True,
        )
    if not records:
        raise RuntimeError("No MI preference pairs passed the reliability filter")
    target.parent.mkdir(parents=True, exist_ok=True)
    diagnostic = target.with_suffix(".diagnostics.json")
    diagnostic.write_text(json.dumps(records, indent=2))
    rows = []
    for record in records:
        # Rebuilt from the authoritative manifest via task_index, never from
        # a transform/descriptor that might have been baked into the
        # diagnostics JSON itself (there isn't one -- see pair_construction.py).
        task = cfg.train_tasks[record["task_index"]]
        prefix = messages(task.transform)
        rows.append({
            "chosen": prefix + [{"role": "assistant", "content": record["chosen_sequence"]}],
            "rejected": prefix + [{"role": "assistant", "content": record["rejected_sequence"]}],
        })
    tmp = target.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
    tmp.replace(target)
    total_seconds = time.perf_counter() - milestone_started
    print(
        f"[ORPT pairs m={milestone}] complete: pairs={len(records)}, "
        f"tasks={total_tasks}, elapsed={total_seconds / 60:.1f}m, output={target}",
        flush=True,
    )
    return target


def train_orpt_milestone(cfg: ExperimentConfig, milestone: int) -> Path:
    final = cfg.orpt_checkpoint_dir(milestone)
    if checkpoint_ready(final):
        return final
    bolt = cfg.milestone_checkpoint_dir(milestone)
    if not checkpoint_ready(bolt):
        raise RuntimeError(f"Missing BOLT checkpoint: {bolt}")
    pairs = build_orpt_pairs(cfg, milestone)
    if cfg.proposal_source == "random":
        final.mkdir(parents=True, exist_ok=True)
        (final / "RANDOM_PROPOSAL_MARKER").write_text(f"pairs={pairs}\n")
        return final
    directory = cfg.bolt_root / FINE_TUNING_DIR
    recipe = directory / "torchtune_config" / cfg.orpt_torchtune_config
    check_training_size(pairs, recipe, cfg.orpt_epochs)
    out = cfg.checkpoints_dir / f"ORPT-{milestone}"
    _run([tune_executable(), "run", "--nnodes", "1", "--nproc_per_node", "1",
          "--master-port", str(torchrun_master_port(cfg)), cfg.orpt_torchtune_recipe,
          "--config", f"torchtune_config/{cfg.orpt_torchtune_config}", f"output_dir={out}",
          f"dataset.data_files={pairs}", f"checkpointer.checkpoint_dir={bolt}",
          f"tokenizer.path={cfg.base_checkpoint_dir}/vocab.json",
          f"tokenizer.merges_file={cfg.base_checkpoint_dir}/merges.txt", f"epochs={cfg.orpt_epochs}",
          f"loss.beta={cfg.orpt_beta}", f"optimizer.lr={cfg.orpt_lr}", "seed=42",
          f"metric_logger.log_dir={cfg.tensorboard_dir / f'ORPT-{milestone}'}"], directory, cfg)
    materialize_hf_checkpoint(out, cfg.orpt_epochs - 1, cfg.base_checkpoint_dir)
    if not checkpoint_ready(final):
        raise RuntimeError(f"DPO did not produce {final}")
    cleanup_intermediate_epochs(out, cfg.orpt_epochs - 1)
    return final
