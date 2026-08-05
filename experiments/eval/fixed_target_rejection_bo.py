"""(2.2) Build a *fixed-size* pool via real rejection sampling (no synthetic
top-up), for each of several target pool sizes, then run real BO on it --
comparing models on equal footing at a chosen initialization-pool size.

If a task's already-generated raw proposals don't contain `target` feasible
unique sequences, that task is skipped for that target (no on-demand extra
sampling) and counted in a coverage_rate metric -- see
common.build_bo_pool(target=...).

Reports coverage separately from BO performance: an arm/milestone that only
clears a given target for a few tasks is a real finding (a naive-DPO arm
paying for its own constraint violations in reduced coverage), not something
to silently average away.

Reads raw generations via common.raw_dir_for(spec, task_set) -- run
generate_raw_proposals.py first for any (arm, milestone, task_set) not yet
covered. Each covered (task, target) launches one real LOLBO run -- start
with --limit-tasks and one target size before a full sweep.

Usage (run from the BOLT repo root):
    python experiments/eval/fixed_target_rejection_bo.py \\
        --config peptide_experiment/configs/peptide_poc20_bolt.yaml \\
        --manifest experiments/eval/manifests/poc20_four_arm.yaml \\
        --target-pool-sizes 10 --limit-tasks 1
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

DEFAULT_TARGET_POOL_SIZES = [10, 20, 50]

COVERAGE_FIELDS = [
    "arm", "milestone", "task_set", "target_pool_size",
    "n_tasks", "n_ran_bo", "coverage_rate", "mean_best_feasible_incumbent",
    "mean_rejection_rate",
]
PER_TASK_FIELDS = [
    "arm", "milestone", "task_set", "task_idx", "target_pool_size",
    "draws_used", "rejection_rate", "best_feasible_incumbent", "bo_calls", "best_mic",
]
SUMMARY_FIELDS = ["arm", "milestone", "task_set", "target_pool_size", "bo_calls", "n_tasks_ran_bo", "mean_best_mic"]


def run_for_spec_task_set_target(
    cfg: ExperimentConfig, spec: ModelSpec, task_set: str, target: int, task_ids: list[int],
) -> tuple[dict, list[dict]]:
    raw_dir = raw_dir_for(spec, task_set)
    work_dir = spec.run_dir / "eval_fixed_target_bo" / task_set / f"{spec.arm}-{spec.milestone}__target{target}"
    run_id = f"fixedtarget-{task_set}-{spec.arm}-{spec.milestone}-t{target}"

    n_ran_bo = 0
    rejection_rates: list[float] = []
    feasible_incumbents: list[float] = []
    bo_rows: list[dict] = []
    for task_idx in task_ids:
        built = build_bo_pool(cfg, task_idx, raw_dir, work_dir, target=target)
        if built is None:
            continue
        init_path, scores_path, pool_size, draws_used = built
        rejection_rate = 1 - pool_size / draws_used if draws_used else None
        if rejection_rate is not None:
            rejection_rates.append(rejection_rate)
        cfg.init_size = pool_size  # run_bo reads cfg.init_size; must match target exactly
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
                "target_pool_size": target, "draws_used": draws_used, "rejection_rate": rejection_rate,
                "best_feasible_incumbent": best_feasible_incumbent,
                "bo_calls": bo_calls, "best_mic": _best_mic_at_k(csv_path, pool_size, bo_calls),
            })

    coverage_row = {
        "arm": spec.arm, "milestone": spec.milestone, "task_set": task_set, "target_pool_size": target,
        "n_tasks": len(task_ids), "n_ran_bo": n_ran_bo,
        "coverage_rate": n_ran_bo / len(task_ids) if task_ids else None,
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
    target_pool_sizes: list[int], limit_tasks: int | None,
) -> tuple[list[dict], list[dict]]:
    coverage_rows, bo_rows = [], []
    for spec in specs:
        for task_set in task_set_names:
            if not raw_dir_for(spec, task_set).exists():
                print(f"[fixed_target_rejection_bo] {spec.arm}-{spec.milestone}/{task_set}: "
                      f"no raw generations, run generate_raw_proposals.py first, skipping")
                continue
            task_ids = task_indices(cfg, task_set)
            if limit_tasks is not None:
                task_ids = task_ids[:limit_tasks]
            for target in target_pool_sizes:
                coverage_row, task_bo_rows = run_for_spec_task_set_target(cfg, spec, task_set, target, task_ids)
                coverage_rows.append(coverage_row)
                bo_rows.extend(task_bo_rows)
    return coverage_rows, bo_rows


def summarize_bo_rows(bo_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    for row in bo_rows:
        if row["best_mic"] is None:
            continue
        key = (row["arm"], row["milestone"], row["task_set"], row["target_pool_size"], row["bo_calls"])
        groups.setdefault(key, []).append(row["best_mic"])
    summary = []
    for (arm, milestone, task_set, target, bo_calls), values in sorted(groups.items()):
        summary.append({
            "arm": arm, "milestone": milestone, "task_set": task_set, "target_pool_size": target,
            "bo_calls": bo_calls, "n_tasks_ran_bo": len(values), "mean_best_mic": sum(values) / len(values),
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-sets", default="heldout")
    parser.add_argument(
        "--target-pool-sizes",
        default=",".join(str(t) for t in DEFAULT_TARGET_POOL_SIZES),
        help="Comma-separated fixed init-pool sizes to compare models at",
    )
    parser.add_argument("--limit-tasks", type=int, default=None, help="Smoke-test: first N tasks per task_set only")
    parser.add_argument("--out-dir", default=None, help="Default: experiments/eval/results/<manifest stem>")
    args = parser.parse_args()

    cfg = load_config(args.config)
    specs = load_manifest(args.manifest)
    task_set_names = [t.strip() for t in args.task_sets.split(",") if t.strip()]
    target_pool_sizes = sorted(int(t.strip()) for t in args.target_pool_sizes.split(",") if t.strip())

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else Path(__file__).resolve().parent / "results" / Path(args.manifest).stem
    )

    coverage_rows, bo_rows = compute_rows(cfg, specs, task_set_names, target_pool_sizes, args.limit_tasks)
    write_csv(coverage_rows, out_dir / "fixed_target_bo_coverage.csv", fieldnames=COVERAGE_FIELDS)
    write_csv(bo_rows, out_dir / "per_task_fixed_target_bo.csv", fieldnames=PER_TASK_FIELDS)
    write_csv(summarize_bo_rows(bo_rows), out_dir / "summary_fixed_target_bo.csv", fieldnames=SUMMARY_FIELDS)


if __name__ == "__main__":
    main()
