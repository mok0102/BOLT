"""Proposal-level performance vs. pool size, no BO -- the compute engine
behind fig:fewshot (paper/experiments.tex sec:main-results, "Initialization
and few-shot proposal quality").

For each (arm, milestone) and each n_proposals cutoff, reports (a) the
incumbent -- best objective among the first n_proposals feasible proposals --
and (b) the rejection ratio needed to draw that many, i.e. how many raw
(possibly infeasible or duplicate) generations it took to accumulate
n_proposals feasible unique candidates.

Separate rows per task_set (trainset / heldout) so trained-task and held-out
performance can be reported independently.

Reads the raw jsonls generate_raw.py wrote. Scores are recomputed fresh here
rather than read off disk: the raw jsonls carry no scores, and any fixed-size
init pool sitting next to them is the padded/patched version, which is the
wrong quantity for "first n_proposals feasible raw proposals".
"""

from __future__ import annotations

from pathlib import Path

from ..core.arms import ArmSpec, raw_dir_for
from ..core.pools import feasible_pool_with_draw_counts, write_csv

DEFAULT_N_PROPOSALS_CHECKPOINTS = [1, 5, 10, 20, 50]

PER_TASK_FIELDS = [
    "arm", "milestone", "task_set", "task_idx", "n_proposals",
    "n_raw_total", "n_feasible_total", "draws_to_n_proposals", "rejection_rate_at_n_proposals", "incumbent_mic",
]
SUMMARY_FIELDS = [
    "arm", "milestone", "task_set", "n_proposals",
    "n_tasks", "coverage_rate_at_n_proposals", "mean_incumbent_mic", "mean_rejection_rate_at_n_proposals",
]


def compute_rows_for_task(
    dom, cfg, spec: ArmSpec, task_set: str, task_id,
    n_proposals_checkpoints: list[int], raw_dir: Path,
) -> list[dict]:
    n_raw_total = len(list(raw_dir.glob(dom.raw_attempt_glob(task_id))))
    pool = feasible_pool_with_draw_counts(dom, cfg, task_id, raw_dir, max_needed=max(n_proposals_checkpoints))
    candidates = [c for c, _ in pool]
    scored = dom.score_candidates(cfg, task_id, candidates) if candidates else []

    base = {"arm": spec.arm, "milestone": spec.milestone, "task_set": task_set, "task_idx": task_id}
    rows = []
    for n_proposals in n_proposals_checkpoints:
        if len(pool) < n_proposals:
            rows.append({
                **base, "n_proposals": n_proposals, "n_raw_total": n_raw_total, "n_feasible_total": len(pool),
                "draws_to_n_proposals": None, "rejection_rate_at_n_proposals": None, "incumbent_mic": None,
            })
            continue
        draws_to_n_proposals = pool[n_proposals - 1][1]
        values = [value for _, value in scored[:n_proposals]]
        rows.append({
            **base, "n_proposals": n_proposals, "n_raw_total": n_raw_total, "n_feasible_total": len(pool),
            "draws_to_n_proposals": draws_to_n_proposals,
            "rejection_rate_at_n_proposals": 1 - n_proposals / draws_to_n_proposals,
            "incumbent_mic": -max(values) if values else None,
        })
    return rows


def compute_rows(dom, cfg, specs: list[ArmSpec], task_sets: list[str], n_proposals_checkpoints: list[int]) -> list[dict]:
    rows = []
    for spec in specs:
        for task_set in task_sets:
            raw_dir = raw_dir_for(spec, task_set)
            if not raw_dir.exists():
                print(f"[eval2.incumbent] {spec.arm}-{spec.milestone}/{task_set}: no raw generations at "
                      f"{raw_dir}, run generate_raw first, skipping")
                continue
            for task_id in dom.task_indices(cfg, task_set):
                rows.extend(compute_rows_for_task(dom, cfg, spec, task_set, task_id, n_proposals_checkpoints, raw_dir))
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["arm"], row["milestone"], row["task_set"], row["n_proposals"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (arm, milestone, task_set, n_proposals), group_rows in sorted(groups.items()):
        sufficient = [r for r in group_rows if r["incumbent_mic"] is not None]
        summary.append({
            "arm": arm, "milestone": milestone, "task_set": task_set, "n_proposals": n_proposals,
            "n_tasks": len(group_rows),
            "coverage_rate_at_n_proposals": len(sufficient) / len(group_rows) if group_rows else None,
            "mean_incumbent_mic": (
                sum(r["incumbent_mic"] for r in sufficient) / len(sufficient) if sufficient else None
            ),
            "mean_rejection_rate_at_n_proposals": (
                sum(r["rejection_rate_at_n_proposals"] for r in sufficient) / len(sufficient) if sufficient else None
            ),
        })
    return summary


def resolve_checkpoints(cfg, requested: list[int] | None) -> list[int]:
    if requested:
        return sorted(requested)
    if cfg.table_k_checkpoints:
        return sorted(cfg.table_k_checkpoints)
    print(f"[eval2.incumbent] no --n-proposals-checkpoints given and cfg.table_k_checkpoints is unset, "
          f"falling back to {DEFAULT_N_PROPOSALS_CHECKPOINTS}")
    return DEFAULT_N_PROPOSALS_CHECKPOINTS


def run(dom, cfg, specs: list[ArmSpec], task_sets: list[str], out_dir: Path, n_proposals_checkpoints: list[int] | None = None) -> None:
    checkpoints = resolve_checkpoints(cfg, n_proposals_checkpoints)
    rows = compute_rows(dom, cfg, specs, task_sets, checkpoints)
    write_csv(rows, out_dir / "per_task_incumbent_vs_pool_size.csv", fieldnames=PER_TASK_FIELDS)
    write_csv(summarize(rows), out_dir / "summary_incumbent_vs_pool_size.csv", fieldnames=SUMMARY_FIELDS)
