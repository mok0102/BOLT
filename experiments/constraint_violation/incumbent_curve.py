"""Fine-grained incumbent (best-found-so-far) curve vs. oracle-call count,
from the already-completed rejection-sampled BO runs
(run_rejection_sampled_bo.py's constraint_violation_bo/ output). Unlike
summarize_rejection_sampled_bo.py -- which only reports best-MIC at a
handful of sparse checkpoints (k in {1,100,200,500,1000,5000}, chosen for
the by-milestone comparison in plot_rejection_sampled_bo.py) -- this reads
each task's full per-acquisition-step trajectory CSV (already on disk, nrow
= pool_size + up to oracle_budget) and evaluates a running max at a fine k
grid, so plot_incumbent_curve.py can draw a real BO convergence curve
instead of a 6-point polyline. No new BO runs: this is pure post-processing
of data run_rejection_sampled_bo.py already produced.

x-axis convention (k = additional oracle calls *after* each task's own
variable-size real-rejection-sampled init pool, not total calls) matches
plot_rejection_sampled_bo.py's fig1/fig2 -- see that script's docstring for
why a total-budget (pool_size + k) x-axis was considered and rejected (pool
size varies per task, not just per arm, so it can't be aligned across tasks
for a clean average). plot_incumbent_curve.py annotates each arm's mean
pool size instead, so the reader isn't misled about total budget consumed.

Reuses peptide_experiment.aggregate._best_mic_at_k's row-index formula
(min(pool_size + k, len(df))) for consistency with every other "best MIC at
k" computation in this repo, and summarize_rejection_sampled_bo.read_pool_size
for the real per-task pool size -- just evaluated at many more k's per task,
via one cummax() per task (O(n) once) instead of recomputing .max() over a
slice per checkpoint.

Usage (run from the BOLT repo root):
    python experiments/constraint_violation/incumbent_curve.py \\
        --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml

    python experiments/constraint_violation/incumbent_curve.py \\
        --config peptide_experiment/configs/peptide_100task_orpt_lexicographic.yaml \\
        --arms ORPT --variant-label ORPT-LEX
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

BOLT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOLT_ROOT))

from peptide_experiment.config import ExperimentConfig, load_config  # noqa: E402

from run_trainset_eval import ARM_CHECKPOINT_FN  # noqa: E402
from measure_violation_rate import task_sets  # noqa: E402
from summarize_rejection_sampled_bo import read_pool_size  # noqa: E402

DEFAULT_K_STEP = 50


def k_grid(oracle_budget: int, k_step: int) -> list[int]:
    return list(range(0, oracle_budget + 1, k_step)) if oracle_budget % k_step == 0 else (
        list(range(0, oracle_budget, k_step)) + [oracle_budget]
    )


def compute_task_curve(csv_path: Path, pool_size: int, ks: list[int]) -> list[tuple[int, float]]:
    """Returns [(k, best_mic), ...] for one task, via one cummax() pass."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    running_best_y = df["train_y"].cummax()
    n = len(running_best_y)

    rows = []
    for k in ks:
        row_idx = min(pool_size + k, n)
        if row_idx <= 0:
            continue
        best_y = running_best_y.iloc[row_idx - 1]
        rows.append((k, -best_y))  # train_y = -MIC (maximized); MIC = -train_y, lower is better
    return rows


def compute_rows(
    cfg: ExperimentConfig,
    arm_prefixes: list[str],
    task_set_names: list[str],
    k_step: int,
    variant_label: str | None = None,
) -> list[dict]:
    ks = k_grid(cfg.oracle_budget, k_step)
    sets = task_sets(cfg)
    rows = []

    for milestone in cfg.milestones:
        for arm in arm_prefixes:
            for task_set in task_set_names:
                bo_dir = cfg.run_dir / "constraint_violation_bo" / task_set / f"{arm}-{milestone}"
                if not bo_dir.exists():
                    print(f"[incumbent_curve] {arm}-{milestone}/{task_set}: no output at {bo_dir}, skipping")
                    continue
                for task_idx in sets[task_set]:
                    csv_path = bo_dir / f"task_{task_idx:04d}.csv"
                    pool_size = read_pool_size(bo_dir, task_idx)
                    if not csv_path.exists() or pool_size is None:
                        continue
                    for k, best_mic in compute_task_curve(csv_path, pool_size, ks):
                        rows.append(
                            {
                                "arm": variant_label or arm,
                                "milestone": milestone,
                                "task_set": task_set,
                                "task_idx": task_idx,
                                "k": k,
                                "pool_size": pool_size,
                                "best_mic": best_mic,
                            }
                        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["arm"], row["milestone"], row["task_set"], row["k"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (arm, milestone, task_set, k), group_rows in sorted(groups.items()):
        n = len(group_rows)
        mics = [r["best_mic"] for r in group_rows]
        mean_mic = sum(mics) / n
        variance = sum((v - mean_mic) ** 2 for v in mics) / n
        summary.append(
            {
                "arm": arm,
                "milestone": milestone,
                "task_set": task_set,
                "k": k,
                "n_tasks": n,
                "mean_best_mic": mean_mic,
                "std_best_mic": variance**0.5,
                "mean_pool_size": sum(r["pool_size"] for r in group_rows) / n,
            }
        )
    return summary


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arms", default="BOLT,ORPT")
    parser.add_argument("--task-sets", default="trainset,heldout20")
    parser.add_argument(
        "--k-step",
        type=int,
        default=DEFAULT_K_STEP,
        help=f"Spacing between k values in the fine grid (default {DEFAULT_K_STEP}, i.e. "
        "0, 50, 100, ... up to cfg.oracle_budget).",
    )
    parser.add_argument(
        "--variant-label",
        default=None,
        help="Override the 'arm' column value in the output CSVs (checkpoint/directory lookup "
        "still uses --arms) -- e.g. 'ORPT-LEX', matching measure_violation_rate.py's convention.",
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    arm_prefixes = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arm_prefixes:
        assert a in ARM_CHECKPOINT_FN, f"unknown arm prefix {a!r}, expected one of {list(ARM_CHECKPOINT_FN)}"
    task_set_names = [t.strip() for t in args.task_sets.split(",") if t.strip()]
    for t in task_set_names:
        assert t in ("trainset", "heldout20"), f"unknown task set {t!r}"

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(__file__).resolve().parent / "results" / cfg.experiment_id
    )

    rows = compute_rows(cfg, arm_prefixes, task_set_names, args.k_step, variant_label=args.variant_label)
    write_csv(
        rows,
        out_dir / "per_task_incumbent_curve.csv",
        fieldnames=["arm", "milestone", "task_set", "task_idx", "k", "pool_size", "best_mic"],
    )
    write_csv(
        summarize(rows),
        out_dir / "summary_incumbent_curve.csv",
        fieldnames=["arm", "milestone", "task_set", "k", "n_tasks", "mean_best_mic", "std_best_mic", "mean_pool_size"],
    )


if __name__ == "__main__":
    main()
