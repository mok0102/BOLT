"""Shared, self-contained helpers for experiments/eval/.

Deliberately independent of experiments/constraint_violation/ (no imports
from it) -- eval/ is a fresh package built directly on the core
peptide_experiment/apex_oracle pipeline, not on constraint_violation/'s
extensions. Some small pieces below (similarity(), load_raw_generations())
are equivalent in spirit to functions already proven out in
constraint_violation/measure_violation_rate.py, reimplemented here rather
than imported.

Model identity in this package is a manifest entry -- (arm, milestone,
run_dir[, checkpoint_dir]) -- not the (cfg, arm-literal) resolution
constraint_violation/ uses, so a comparison can freely span models trained
under different experiment_id/run_dir trees (e.g. BOLT from one PoC config,
ORPT-FA from another).
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

BOLT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOLT_ROOT))
sys.path.insert(0, str(BOLT_ROOT / "fine-tuning" / "peptides" / "sampled_output_from_ft"))

from peptide_experiment.config import ExperimentConfig  # noqa: E402
from apex_oracle.refseqs import REFERENCE_SEQUENCE  # noqa: E402
from make_initialization_data import extract_sequence  # noqa: E402
from Levenshtein import distance as edit_distance  # noqa: E402

TASK_SETS = ("trainset", "heldout")


def similarity(seq: str, reference: str) -> float:
    """Same edit-distance-based formula as
    peptide_experiment/steps.py::_ensure_constraint_feasible."""
    length = len(reference)
    return (length - edit_distance(seq, reference)) / length


def load_raw_generations(task_dir: Path, task_idx: int) -> list[str]:
    """Raw, order-preserved, non-deduped sequences from every
    task_<idx>_sampled_attempt*.jsonl in task_dir, in attempt order."""
    attempt_files = sorted(
        task_dir.glob(f"task_{task_idx:04d}_sampled_attempt*.jsonl"),
        key=lambda p: int(p.stem.rsplit("attempt", 1)[1]),
    )
    sequences: list[str] = []
    for f in attempt_files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for answer in record.get("generated_answers", []):
                if answer is None:
                    continue
                seq = extract_sequence(str(answer))
                if seq:
                    sequences.append(seq)
    return sequences


@dataclass
class ModelSpec:
    arm: str
    milestone: int
    run_dir: Path
    checkpoint_dir: Path | None = None  # only read by generate_raw_proposals.py


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else BOLT_ROOT / path


def load_manifest(path: Path) -> list[ModelSpec]:
    path = Path(path)
    if not path.is_absolute():
        path = BOLT_ROOT / path
    with open(path) as f:
        raw = yaml.safe_load(f)

    specs: list[ModelSpec] = []
    seen: set[tuple[str, int]] = set()
    for entry in raw["models"]:
        spec = ModelSpec(
            arm=entry["arm"],
            milestone=int(entry["milestone"]),
            run_dir=_resolve(entry["run_dir"]),
            checkpoint_dir=_resolve(entry["checkpoint_dir"]) if entry.get("checkpoint_dir") else None,
        )
        key = (spec.arm, spec.milestone)
        if key in seen:
            raise ValueError(f"duplicate (arm, milestone) in manifest {path}: {key}")
        seen.add(key)
        specs.append(spec)
    return specs


def raw_dir_for(spec: ModelSpec, task_set: str) -> Path:
    """eval/'s own raw-generation directory convention -- parallel to, but
    independent of, constraint_violation/'s trainset_eval/heldout20/init_only
    naming, so the two pipelines never share or collide on output."""
    assert task_set in TASK_SETS, f"unknown task_set {task_set!r}, expected one of {TASK_SETS}"
    return spec.run_dir / "eval_raw" / task_set / f"{spec.arm}-{spec.milestone}"


def task_indices(cfg: ExperimentConfig, task_set: str) -> list[int]:
    """trainset: the fixed subset every milestone's checkpoint has already
    been trained on (range(min(milestones))) -- apples-to-apples across
    milestones, distinct from cfg.train_task_range()'s full cumulative
    max(milestones) range. heldout: cfg's held-out task list (respects
    heldout_tasks_override, e.g. the 5-task PoC subset)."""
    if task_set == "trainset":
        return list(range(min(cfg.milestones)))
    if task_set == "heldout":
        return list(cfg.heldout_tasks("heldout20"))
    raise ValueError(f"unknown task_set {task_set!r}, expected one of {TASK_SETS}")


def feasible_pool_with_draw_counts(
    cfg: ExperimentConfig,
    task_idx: int,
    raw_dir: Path,
    max_needed: int | None = None,
) -> list[tuple[str, int]]:
    """One pass over load_raw_generations(): dedup + similarity-filter,
    returning (sequence, 1-indexed draw_index) for each accepted sequence,
    stopping early once max_needed are collected (None = collect all)."""
    reference = REFERENCE_SEQUENCE[task_idx]
    sequences = load_raw_generations(raw_dir, task_idx)
    pool: list[tuple[str, int]] = []
    seen: set[str] = set()
    for draw_idx, seq in enumerate(sequences, start=1):
        if seq in seen:
            continue
        seen.add(seq)
        if similarity(seq, reference) >= cfg.similarity_threshold:
            pool.append((seq, draw_idx))
            if max_needed is not None and len(pool) >= max_needed:
                break
    return pool


def bo_k_checkpoints(cfg: ExperimentConfig) -> list[int]:
    """Oracle-call checkpoints at which to report BO performance -- cfg's
    table_k_checkpoints plus the full oracle_budget itself."""
    return sorted(set(cfg.table_k_checkpoints) | {cfg.oracle_budget})


def read_pool_size(bo_dir: Path, task_idx: int) -> int:
    path = bo_dir / f"task_{task_idx:04d}_init.txt"
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def read_best_feasible_incumbent(bo_dir: Path, task_idx: int) -> float | None:
    """Best (lowest) MIC among the built init pool itself, i.e. the
    generation-time incumbent before any BO acquisition."""
    path = bo_dir / f"task_{task_idx:04d}_scores.csv"
    if not path.exists():
        return None
    scores = [float(x) for x in path.read_text().splitlines() if x.strip()]
    return -max(scores) if scores else None


def build_bo_pool(
    cfg: ExperimentConfig,
    task_idx: int,
    raw_dir: Path,
    work_dir: Path,
    target: int | None = None,
    min_feasible: int = 5,
) -> tuple[Path, Path, int, int] | None:
    """Build a BO init pool from raw generations, real rejection sampling
    only (no synthetic top-up/padding). Idempotent on existing
    task_XXXX_init.txt/_scores.csv in work_dir (draws_used is still
    recomputed on the idempotent path -- it's a cheap raw-jsonl rescan, no
    apex_wrapper call).

    target=None: use the full real-rejection-sampled feasible pool, skip if
    fewer than min_feasible survive (the "fixed budget" mode). draws_used is
    every raw sequence generated (the whole sampling budget was consumed).
    target=<int>: require exactly `target` feasible unique sequences, skip
    (return None) if fewer are available -- no on-demand extra sampling (the
    "fixed target" mode). draws_used is the raw draw index at which the
    `target`-th feasible sequence appeared.

    Returns (init_path, scores_path, pool_size, draws_used); rejection_rate
    = 1 - pool_size / draws_used is left to callers (they already vary in
    what else they log alongside it).
    """
    from apex_oracle import apex_wrapper

    init_path = work_dir / f"task_{task_idx:04d}_init.txt"
    scores_path = work_dir / f"task_{task_idx:04d}_scores.csv"

    pool = feasible_pool_with_draw_counts(cfg, task_idx, raw_dir, max_needed=target)
    seqs = [s for s, _ in pool]
    floor = target if target is not None else min_feasible
    if len(seqs) < floor:
        return None
    if target is not None:
        seqs = seqs[:target]
        draws_used = pool[target - 1][1]
    else:
        draws_used = len(load_raw_generations(raw_dir, task_idx))

    if init_path.exists() and scores_path.exists():
        return init_path, scores_path, read_pool_size(work_dir, task_idx), draws_used

    scores = list(-apex_wrapper(seqs)[:, 0])
    work_dir.mkdir(parents=True, exist_ok=True)
    init_path.write_text("\n".join(seqs) + "\n")
    scores_path.write_text("\n".join(f"{s:.8f}" for s in scores) + "\n")
    return init_path, scores_path, len(seqs), draws_used


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")
