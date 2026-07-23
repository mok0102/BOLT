"""Summarize run_rejection_sampled_bo.py's output: per (arm, milestone,
task_set), what fraction of tasks had enough feasible unique candidates for
real rejection sampling to even attempt BO ("coverage", plus the actual
feasible-pool-size distribution), and, among the tasks that did run, the
best objective (MIC) found within each oracle-call checkpoint.

Reports coverage/pool-size separately from BO performance because they
answer different questions: coverage says how often a reject-and-discard
policy can start optimizing at all, and with how much initial diversity (an
arm/milestone that only clears run_rejection_sampled_bo.py's --min-feasible
floor for a handful of tasks, or only with tiny pools, is a real failure,
not noise); best-MIC-at-k says how well BO does once it does start, given
whatever pool size that task actually got (init pool size is NOT fixed
across tasks/arms by design -- see run_rejection_sampled_bo.py's docstring
for why). Folding skipped tasks into the objective average would hide the
coverage story inside a smaller sample rather than reporting it.

Best-MIC-at-k formula reused as-is from
peptide_experiment/aggregate.py::_best_mic_at_k (train_y is the running
acquisition order, MIC = -train_y, lower is better; "best within k oracle
calls" = running max of train_y over rows [0 : init_size + k]) -- init_size
here is READ PER TASK from the persisted task_XXXX_init.txt line count, not
taken from cfg, since every task's real rejection-sampled pool is a
different size.

Usage (run from the BOLT repo root):
    python experiments/constraint_violation/summarize_rejection_sampled_bo.py \\
        --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml \\
        --out-dir experiments/constraint_violation/results/peptide_100task_orpt_beta0.25
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

BOLT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOLT_ROOT))

from peptide_experiment.config import ExperimentConfig, load_config  # noqa: E402
from peptide_experiment.aggregate import _best_mic_at_k  # noqa: E402

from run_trainset_eval import ARM_CHECKPOINT_FN  # noqa: E402
from measure_violation_rate import task_dir_for, task_sets  # noqa: E402
from run_rejection_sampled_bo import real_rejection_sample  # noqa: E402


def k_checkpoints(cfg: ExperimentConfig) -> list[int]:
    return sorted(set(cfg.table_k_checkpoints) | {cfg.oracle_budget})


def read_pool_size(bo_dir: Path, task_idx: int) -> int | None:
    init_path = bo_dir / f"task_{task_idx:04d}_init.txt"
    if not init_path.exists():
        return None
    return sum(1 for line in init_path.read_text().splitlines() if line.strip())


def read_best_feasible_incumbent(bo_dir: Path, task_idx: int) -> float | None:
    """Best (lowest) MIC among the real-rejection-sampled feasible init pool
    itself -- i.e. the generation-time incumbent, before any BO acquisition
    steps. build_rejection_sampled_init() already writes this pool's scores
    (train_y = -MIC, same convention as trajectories_csv) to
    task_XXXX_scores.csv; this just reads them back rather than
    recomputing anything."""
    scores_path = bo_dir / f"task_{task_idx:04d}_scores.csv"
    if not scores_path.exists():
        return None
    scores = [float(line) for line in scores_path.read_text().splitlines() if line.strip()]
    if not scores:
        return None
    return -max(scores)


def compute_rows(
    cfg: ExperimentConfig,
    arm_prefixes: list[str],
    task_set_names: list[str],
    variant_label: str | None = None,
):
    """variant_label, if given, overrides the 'arm' value written to output rows (directory
    lookup / checkpoint resolution still uses the real arm prefix) -- see
    measure_violation_rate.py's compute_rows() for why: lets a differently-trained checkpoint
    under the same arm name (e.g. a lexicographic-pairing ORPT variant in its own
    experiment_id/run_dir) be told apart once its CSV is concatenated with another
    experiment's results."""
    sets = task_sets(cfg)
    coverage_rows, bo_rows = [], []

    for milestone in cfg.milestones:
        for arm in arm_prefixes:
            for task_set in task_set_names:
                raw_task_dir = task_dir_for(cfg, task_set, arm, milestone)
                bo_dir = cfg.run_dir / "constraint_violation_bo" / task_set / f"{arm}-{milestone}"
                task_indices = sets[task_set]
                if not raw_task_dir.exists():
                    continue

                pool_sizes, feasible_incumbents, n_ran_bo = [], [], 0
                for task_idx in task_indices:
                    feasible_pool_size = len(real_rejection_sample(cfg, task_idx, raw_task_dir))
                    pool_sizes.append(feasible_pool_size)

                    csv_path = bo_dir / f"task_{task_idx:04d}.csv"
                    pool_size = read_pool_size(bo_dir, task_idx)
                    if not csv_path.exists() or pool_size is None:
                        continue
                    n_ran_bo += 1
                    best_feasible_incumbent = read_best_feasible_incumbent(bo_dir, task_idx)
                    if best_feasible_incumbent is not None:
                        feasible_incumbents.append(best_feasible_incumbent)
                    for k in k_checkpoints(cfg):
                        best_mic = _best_mic_at_k(csv_path, pool_size, k)
                        bo_rows.append(
                            {
                                "arm": variant_label or arm,
                                "milestone": milestone,
                                "task_set": task_set,
                                "task_idx": task_idx,
                                "pool_size": pool_size,
                                "best_feasible_incumbent": best_feasible_incumbent,
                                "k": k,
                                "best_mic": best_mic,
                            }
                        )

                coverage_rows.append(
                    {
                        "arm": variant_label or arm,
                        "milestone": milestone,
                        "task_set": task_set,
                        "n_tasks": len(task_indices),
                        "n_ran_bo": n_ran_bo,
                        "coverage_rate": n_ran_bo / len(task_indices) if task_indices else None,
                        "min_pool_size": min(pool_sizes) if pool_sizes else None,
                        "mean_pool_size": sum(pool_sizes) / len(pool_sizes) if pool_sizes else None,
                        "max_pool_size": max(pool_sizes) if pool_sizes else None,
                        "mean_best_feasible_incumbent": (
                            sum(feasible_incumbents) / len(feasible_incumbents)
                            if feasible_incumbents
                            else None
                        ),
                    }
                )

    return coverage_rows, bo_rows


def summarize_bo_rows(bo_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    for row in bo_rows:
        if row["best_mic"] is None:
            continue
        key = (row["arm"], row["milestone"], row["task_set"], row["k"])
        groups.setdefault(key, []).append(row["best_mic"])
    summary = []
    for (arm, milestone, task_set, k), values in sorted(groups.items()):
        summary.append(
            {
                "arm": arm,
                "milestone": milestone,
                "task_set": task_set,
                "k": k,
                "n_tasks_ran_bo": len(values),
                "mean_best_mic": sum(values) / len(values),
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
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--variant-label",
        default=None,
        help="Override the 'arm' column value in the output CSVs (checkpoint lookup still uses "
        "--arms) -- e.g. 'ORPT-LEX' for a lexicographic-pairing ORPT run in its own "
        "experiment_id, so its results stay distinguishable from another experiment's 'ORPT' "
        "rows once the two CSVs are concatenated for comparison.",
    )
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

    coverage_rows, bo_rows = compute_rows(cfg, arm_prefixes, task_set_names, variant_label=args.variant_label)
    write_csv(
        coverage_rows,
        out_dir / "rejection_sampled_bo_coverage.csv",
        fieldnames=[
            "arm", "milestone", "task_set", "n_tasks", "n_ran_bo", "coverage_rate",
            "min_pool_size", "mean_pool_size", "max_pool_size", "mean_best_feasible_incumbent",
        ],
    )
    write_csv(
        bo_rows,
        out_dir / "per_task_rejection_sampled_bo.csv",
        fieldnames=[
            "arm", "milestone", "task_set", "task_idx", "pool_size",
            "best_feasible_incumbent", "k", "best_mic",
        ],
    )
    write_csv(
        summarize_bo_rows(bo_rows),
        out_dir / "summary_rejection_sampled_bo.csv",
        fieldnames=["arm", "milestone", "task_set", "k", "n_tasks_ran_bo", "mean_best_mic"],
    )


if __name__ == "__main__":
    main()
