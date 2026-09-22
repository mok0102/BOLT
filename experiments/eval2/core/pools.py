"""Building a BO init pool out of raw LLM generations, by real rejection
sampling only -- no synthetic top-up or padding.

Domain-neutral plumbing: every candidate-specific decision (parse, dedup,
feasibility, score, on-disk format) is delegated to the domain module passed
in as `dom`, so adding a second domain later means passing a different module
here, not editing this file.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BuiltPool:
    pool_size: int
    draws_used: int
    run_bo_kwargs: dict  # forwarded as **kwargs into the domain's run_bo()


def feasible_pool_with_draw_counts(dom, cfg, task_id, raw_dir: Path, max_needed: int | None = None) -> list[tuple]:
    """One pass over the raw generations: dedup then feasibility-filter,
    returning (candidate, 1-indexed draw_index) per accepted candidate.
    Stops early once max_needed are collected (None = collect all).

    The draw index is what makes a rejection rate computable later: it says
    how many raw draws the sampler had to burn to reach this candidate.
    """
    candidates = dom.load_raw_candidates(raw_dir, task_id)
    pool: list[tuple] = []
    seen: set = set()
    for draw_idx, candidate in enumerate(candidates, start=1):
        key = dom.dedup_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        if dom.is_feasible(cfg, task_id, candidate):
            pool.append((candidate, draw_idx))
            if max_needed is not None and len(pool) >= max_needed:
                break
    return pool


def bo_k_checkpoints(cfg) -> list[int]:
    """Oracle-call checkpoints at which to report BO performance: cfg's
    table_k_checkpoints plus the full oracle_budget itself."""
    return sorted(set(cfg.table_k_checkpoints or []) | {cfg.oracle_budget})


def build_bo_pool(
    dom,
    cfg,
    task_id,
    raw_dir: Path,
    work_dir: Path,
    target: int | None = None,
    min_feasible: int = 5,
) -> BuiltPool | None:
    """Idempotent on an existing init pool in work_dir (dom.read_existing_pool);
    draws_used is still recomputed on that path, a cheap raw-generations
    rescan with no oracle call.

    target=None ("fixed budget"): use the full feasible pool, skip if fewer
    than min_feasible survive. draws_used is every raw candidate generated --
    the whole sampling budget was consumed.
    target=<int> ("fixed target"): require exactly `target` feasible unique
    candidates, return None if fewer are available -- no on-demand extra
    sampling. draws_used is the draw index at which the target-th feasible
    candidate appeared.

    rejection_rate = 1 - pool_size / draws_used is left to callers, which
    already vary in what else they log alongside it.
    """
    pool = feasible_pool_with_draw_counts(dom, cfg, task_id, raw_dir, max_needed=target)
    candidates = [c for c, _ in pool]
    floor = target if target is not None else min_feasible
    if len(candidates) < floor:
        return None
    if target is not None:
        candidates = candidates[:target]
        draws_used = pool[target - 1][1]
    else:
        draws_used = len(dom.load_raw_candidates(raw_dir, task_id))

    existing = dom.read_existing_pool(work_dir, task_id)
    if existing is not None:
        pool_size, run_bo_kwargs = existing
        return BuiltPool(pool_size=pool_size, draws_used=draws_used, run_bo_kwargs=run_bo_kwargs)

    scored = dom.score_candidates(cfg, task_id, candidates)
    run_bo_kwargs = dom.write_init_pool(work_dir, task_id, scored)
    return BuiltPool(pool_size=len(scored), draws_used=draws_used, run_bo_kwargs=run_bo_kwargs)


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")
