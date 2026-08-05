"""(2.3) Run real BO seeded by exactly whatever survives real rejection
sampling on a *fixed sampling budget* (whatever generate_raw_proposals.py
already produced) -- no fixed target pool size, no synthetic top-up/padding,
no truncation. Pool size varies task-to-task and arm-to-arm by design: an
arm whose proposals violate the similarity constraint more ends up with a
smaller real feasible pool, and that is itself part of what's being
measured, not something to normalize away. Only a small absolute floor is
enforced (--min-feasible) to dodge the LOLBO trust-region hang on a
(near-)empty init pool.

Reports coverage (how often, and with how much pool-size diversity, a
reject-and-discard policy can even start optimizing) separately from BO
performance (how well it does once it does start) -- folding skipped tasks
into the objective average would hide the coverage story inside a smaller
sample.

Reads raw generations via common.raw_dir_for(spec, task_set) -- run
generate_raw_proposals.py first for any (arm, milestone, task_set) not yet
covered. Each covered task launches one real LOLBO run -- start with
--limit-tasks before a full sweep.

Usage (run from the BOLT repo root):
    python experiments/eval/fixed_budget_rejection_bo.py \\
        --config peptide_experiment/configs/peptide_poc20_bolt.yaml \\
        --manifest experiments/eval/manifests/poc20_four_arm.yaml \\
        --limit-tasks 1
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import (
    ModelSpec,
    bo_k_checkpoints,
    build_bo_pool,
    load_manifest,
    raw_dir_for,
    read_best_feasible_incumbent,
    task_indices,
    write_csv,
)

from peptide_experiment.aggregate import _best_mic_at_k
from peptide_experiment.config import ExperimentConfig, load_config
from peptide_experiment.steps import run_bo

DEFAULT_MIN_FEASIBLE = 5

COVERAGE_FIELDS = [
    "arm", "milestone", "task_set", "n_tasks", "n_ran_bo", "coverage_rate",
    "min_pool_size", "mean_pool_size", "max_pool_size", "mean_best_feasible_incumbent",
    "mean_rejection_rate",
]
PER_TASK_FIELDS = [
    "arm", "milestone", "task_set", "task_idx", "pool_size", "draws_used", "rejection_rate",
    "best_feasible_incumbent", "bo_calls", "best_mic",
]
SUMMARY_FIELDS = ["arm", "milestone", "task_set", "bo_calls", "n_tasks_ran_bo", "mean_best_mic"]


def run_for_spec_task_set(
    cfg: ExperimentConfig, spec: ModelSpec, task_set: str, task_ids: list[int], min_feasible: int,
) -> tuple[dict, list[dict]]:
    raw_dir = raw_dir_for(spec, task_set)
    work_dir = spec.run_dir / "eval_fixed_budget_bo" / task_set / f"{spec.arm}-{spec.milestone}"
    run_id = f"fixedbudget-{task_set}-{spec.arm}-{spec.milestone}"

    n_ran_bo = 0
    pool_sizes: list[int] = []
    rejection_rates: list[float] = []
    feasible_incumbents: list[float] = []
    bo_rows: list[dict] = []
    for task_idx in task_ids:
        built = build_bo_pool(cfg, task_idx, raw_dir, work_dir, target=None, min_feasible=min_feasible)
        if built is None:
            continue
        init_path, scores_path, pool_size, draws_used = built
        pool_sizes.append(pool_size)
        rejection_rate = 1 - pool_size / draws_used if draws_used else None
        if rejection_rate is not None:
            rejection_rates.append(rejection_rate)
        cfg.init_size = pool_size
        try:
            csv_path = run_bo(cfg, task_idx, work_dir, run_id=run_id, init_path=init_path, scores_path=scores_path)
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"[{run_id} task {task_idx}] FAILED, continuing with rest of sweep: {e}")
            continue
        n_ran_bo += 1
        best_feasible_incumbent = read_best_feasible_incumbent(work_dir, task_idx)
        if best_feasible_incumbent is not None:
            feasible_incumbents.append(best_feasible_incumbent)
        for bo_calls in bo_k_checkpoints(cfg):
            bo_rows.append({
                "arm": spec.arm, "milestone": spec.milestone, "task_set": task_set, "task_idx": task_idx,
                "pool_size": pool_size, "draws_used": draws_used, "rejection_rate": rejection_rate,
                "best_feasible_incumbent": best_feasible_incumbent,
                "bo_calls": bo_calls, "best_mic": _best_mic_at_k(csv_path, pool_size, bo_calls),
            })

    coverage_row = {
        "arm": spec.arm, "milestone": spec.milestone, "task_set": task_set,
        "n_tasks": len(task_ids), "n_ran_bo": n_ran_bo,
        "coverage_rate": n_ran_bo / len(task_ids) if task_ids else None,
        "min_pool_size": min(pool_sizes) if pool_sizes else None,
        "mean_pool_size": sum(pool_sizes) / len(pool_sizes) if pool_sizes else None,
        "max_pool_size": max(pool_sizes) if pool_sizes else None,
        "mean_best_feasible_incumbent": (
            sum(feasible_incumbents) / len(feasible_incumbents) if feasible_incumbents else None
        ),
        "mean_rejection_rate": (
            sum(rejection_rates) / len(rejection_rates) if rejection_rates else None
        ),
    }
    return coverage_row, bo_rows


def compute_rows(
    cfg: ExperimentConfig, specs: list[ModelSpec], task_set_names: list[str],
    min_feasible: int, limit_tasks: int | None,
) -> tuple[list[dict], list[dict]]:
    coverage_rows, bo_rows = [], []
    for spec in specs:
        for task_set in task_set_names:
            if not raw_dir_for(spec, task_set).exists():
                print(f"[fixed_budget_rejection_bo] {spec.arm}-{spec.milestone}/{task_set}: "
                      f"no raw generations, run generate_raw_proposals.py first, skipping")
                continue
            task_ids = task_indices(cfg, task_set)
            if limit_tasks is not None:
                task_ids = task_ids[:limit_tasks]
            coverage_row, task_bo_rows = run_for_spec_task_set(cfg, spec, task_set, task_ids, min_feasible)
            coverage_rows.append(coverage_row)
            bo_rows.extend(task_bo_rows)
    return coverage_rows, bo_rows


def summarize_bo_rows(bo_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    for row in bo_rows:
        if row["best_mic"] is None:
            continue
        key = (row["arm"], row["milestone"], row["task_set"], row["bo_calls"])
        groups.setdefault(key, []).append(row["best_mic"])
    summary = []
    for (arm, milestone, task_set, bo_calls), values in sorted(groups.items()):
        summary.append({
            "arm": arm, "milestone": milestone, "task_set": task_set,
            "bo_calls": bo_calls, "n_tasks_ran_bo": len(values), "mean_best_mic": sum(values) / len(values),
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-sets", default="heldout")
    parser.add_argument("--min-feasible", type=int, default=DEFAULT_MIN_FEASIBLE)
    parser.add_argument("--limit-tasks", type=int, default=None, help="Smoke-test: first N tasks per task_set only")
    parser.add_argument("--out-dir", default=None, help="Default: experiments/eval/results/<manifest stem>")
    args = parser.parse_args()

    cfg = load_config(args.config)
    specs = load_manifest(args.manifest)
    task_set_names = [t.strip() for t in args.task_sets.split(",") if t.strip()]

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else Path(__file__).resolve().parent / "results" / Path(args.manifest).stem
    )

    coverage_rows, bo_rows = compute_rows(cfg, specs, task_set_names, args.min_feasible, args.limit_tasks)
    write_csv(coverage_rows, out_dir / "fixed_budget_bo_coverage.csv", fieldnames=COVERAGE_FIELDS)
    write_csv(bo_rows, out_dir / "per_task_fixed_budget_bo.csv", fieldnames=PER_TASK_FIELDS)
    write_csv(summarize_bo_rows(bo_rows), out_dir / "summary_fixed_budget_bo.csv", fieldnames=SUMMARY_FIELDS)


if __name__ == "__main__":
    main()
