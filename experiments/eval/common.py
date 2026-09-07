"""Shared, self-contained helpers for experiments/eval/.

Domain-generic (see domains.py) -- everything here takes a Domain instance
rather than assuming peptide_experiment/apex_oracle specifically. Deliberately
independent of experiments/constraint_violation/ (no imports from it, and
that directory has since been removed).

Model identity in this package is a manifest entry -- (arm, milestone,
run_dir[, checkpoint_dir]) -- not a (cfg, arm-literal) resolution, so a
comparison can freely span models trained under different
experiment_id/run_dir trees (e.g. BOLT from one config, ORPT-H1 from another).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import yaml

from domains import BuiltPool, Domain

BOLT_ROOT = Path(__file__).resolve().parents[2]


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
    independent of, the now-removed constraint_violation/'s trainset_eval/
    heldout20/init_only naming. Validity of task_set itself is the calling
    domain's responsibility (Domain.task_indices raises on an unknown one)."""
    return spec.run_dir / "eval_raw" / task_set / f"{spec.arm}-{spec.milestone}"


def feasible_pool_with_draw_counts(
    domain: Domain,
    cfg,
    task_id,
    raw_dir: Path,
    max_needed: int | None = None,
) -> list[tuple]:
    """One pass over domain.load_raw_candidates(): dedup (domain.dedup_key)
    + feasibility-filter (domain.is_feasible -- always True for query_plan,
    since that domain has no pre-oracle constraint), returning (candidate,
    1-indexed draw_index) for each accepted candidate, stopping early once
    max_needed are collected (None = collect all)."""
    candidates = domain.load_raw_candidates(raw_dir, task_id)
    pool: list[tuple] = []
    seen: set = set()
    for draw_idx, candidate in enumerate(candidates, start=1):
        key = domain.dedup_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        if domain.is_feasible(cfg, task_id, candidate):
            pool.append((candidate, draw_idx))
            if max_needed is not None and len(pool) >= max_needed:
                break
    return pool


def bo_k_checkpoints(cfg) -> list[int]:
    """Oracle-call checkpoints at which to report BO performance -- cfg's
    table_k_checkpoints plus the full oracle_budget itself. table_k_checkpoints
    may be None (query_plan's own default) -- treated as empty rather than
    erroring, matching query_plan_experiment/aggregate.py's own fallback."""
    return sorted(set(cfg.table_k_checkpoints or []) | {cfg.oracle_budget})


def read_pool_size(bo_dir: Path, task_idx: int) -> int:
    """Peptide-only (used by the MTBO/POGPE/SGPE baseline branches in
    fixed_target_rejection_bo.py, which self-seed via build_mutation_init --
    no query_plan analog exists yet)."""
    path = bo_dir / f"task_{task_idx:04d}_init.txt"
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def read_best_feasible_incumbent(bo_dir: Path, task_idx: int) -> float | None:
    """Best (lowest) MIC among the built init pool itself, i.e. the
    generation-time incumbent before any BO acquisition. Peptide-only, same
    reason as read_pool_size."""
    path = bo_dir / f"task_{task_idx:04d}_scores.csv"
    if not path.exists():
        return None
    scores = [float(x) for x in path.read_text().splitlines() if x.strip()]
    return -max(scores) if scores else None


def build_bo_pool(
    domain: Domain,
    cfg,
    task_id,
    raw_dir: Path,
    work_dir: Path,
    target: int | None = None,
    min_feasible: int = 5,
) -> BuiltPool | None:
    """Build a BO init pool from raw generations, real rejection sampling
    only (no synthetic top-up/padding). Idempotent on an existing init pool
    in work_dir (domain.read_existing_pool) -- draws_used is still
    recomputed on the idempotent path, a cheap raw-generations rescan, no
    oracle call.

    target=None: use the full real-rejection-sampled feasible pool, skip if
    fewer than min_feasible survive (the "fixed budget" mode). draws_used is
    every raw candidate generated (the whole sampling budget was consumed).
    target=<int>: require exactly `target` feasible unique candidates, skip
    (return None) if fewer are available -- no on-demand extra sampling (the
    "fixed target" mode). draws_used is the raw draw index at which the
    `target`-th feasible candidate appeared.

    Returns a BuiltPool (pool_size, draws_used, run_bo_kwargs); rejection_rate
    = 1 - pool_size / draws_used is left to callers (they already vary in
    what else they log alongside it).
    """
    pool = feasible_pool_with_draw_counts(domain, cfg, task_id, raw_dir, max_needed=target)
    candidates = [c for c, _ in pool]
    floor = target if target is not None else min_feasible
    if len(candidates) < floor:
        return None
    if target is not None:
        candidates = candidates[:target]
        draws_used = pool[target - 1][1]
    else:
        draws_used = len(domain.load_raw_candidates(raw_dir, task_id))

    existing = domain.read_existing_pool(work_dir, task_id)
    if existing is not None:
        pool_size, run_bo_kwargs = existing
        return BuiltPool(pool_size=pool_size, draws_used=draws_used, run_bo_kwargs=run_bo_kwargs)

    scored = domain.score_candidates(cfg, task_id, candidates)
    run_bo_kwargs = domain.write_init_pool(work_dir, task_id, scored)
    return BuiltPool(pool_size=len(scored), draws_used=draws_used, run_bo_kwargs=run_bo_kwargs)


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")
