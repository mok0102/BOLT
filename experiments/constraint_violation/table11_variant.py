"""Table-11-style ("few-shot"/init-only) per-task numbers, computed the same
way as peptide_experiment.aggregate.build_table11() but broken out per task
(not summed) and taggable with a --variant-label -- so a checkpoint-family
trained under its own experiment_id/run_dir (e.g. ORPT-LEX, the
lexicographic-pairing DPO variant, which has no BOLT/STBO heldout eval of
its own to compare against) can still be combined with another experiment's
real BOLT/ORPT/STBO numbers into one comparison chart. See plot_table11.py.

build_table11() itself can't be reused directly for this: it always sums
across peptide_experiment.aggregate._arms(cfg), which is BOLT-<m> +
ORPT-<m> + STBO all drawn from the SAME cfg/run_dir -- pointing it at
peptide_100task_orpt_lexicographic's config produces a table where every
BOLT-<m>/STBO column is a 0-with-"missing" warning (that experiment's own
run_dir never ran BOLT/STBO heldout eval, only ORPT), and the real ORPT-<m>
numbers there are literally not distinguishable by column name from another
experiment's naive-DPO "ORPT-<m>" columns. This script computes per-task
rows scoped to just the requested arm prefixes (default BOLT,ORPT) directly
from that experiment's own init_only eval output, reusing
aggregate._best_score_at_k_from_scores (the exact same per-task MIC-at-k
formula Table 11 itself uses) so the numbers match build_table11() exactly
once summed -- just with a task-level breakdown and a renameable arm label.

Usage (run from the BOLT repo root):
    python experiments/constraint_violation/table11_variant.py \\
        --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml

    python experiments/constraint_violation/table11_variant.py \\
        --config peptide_experiment/configs/peptide_100task_orpt_lexicographic.yaml \\
        --arms ORPT --variant-label ORPT-LEX
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

BOLT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOLT_ROOT))

from peptide_experiment.aggregate import _best_score_at_k_from_scores  # noqa: E402
from peptide_experiment.config import ExperimentConfig, load_config  # noqa: E402


def compute_rows(
    cfg: ExperimentConfig,
    arm_prefixes: list[str],
    k_checkpoints: list[int],
    variant_label: str | None = None,
) -> list[dict]:
    rows = []
    tasks = cfg.heldout_tasks("heldout20")
    for milestone in cfg.milestones:
        for arm_prefix in arm_prefixes:
            arm_name = f"{arm_prefix}-{milestone}"
            arm_dir = cfg.heldout_dir("heldout20") / "init_only" / arm_name
            if not arm_dir.exists():
                print(f"[table11_variant] {arm_name}: no output at {arm_dir}, skipping")
                continue
            for task_idx in tasks:
                scores_path = arm_dir / f"task_{task_idx:04d}_scores.csv"
                for k in k_checkpoints:
                    mic = _best_score_at_k_from_scores(scores_path, k)
                    if mic is None:
                        continue
                    rows.append(
                        {
                            "arm": variant_label or arm_prefix,
                            "milestone": milestone,
                            "task_idx": task_idx,
                            "k": k,
                            "mic": mic,
                        }
                    )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    for row in rows:
        key = (row["arm"], row["milestone"], row["k"])
        groups.setdefault(key, []).append(row["mic"])
    summary = []
    for (arm, milestone, k), values in sorted(groups.items()):
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        summary.append(
            {
                "arm": arm,
                "milestone": milestone,
                "k": k,
                "n_tasks": n,
                "mean_mic": mean,
                "std_mic": variance**0.5,
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
    parser.add_argument(
        "--k-checkpoints",
        default=None,
        help="Comma-separated k cutoffs (default: cfg.table_k_checkpoints)",
    )
    parser.add_argument(
        "--variant-label",
        default=None,
        help="Override the 'arm' column value in the output CSVs (directory lookup still uses "
        "--arms) -- e.g. 'ORPT-LEX' so this run's numbers stay distinguishable from another "
        "experiment's 'ORPT' rows once concatenated for comparison.",
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    arm_prefixes = [a.strip() for a in args.arms.split(",") if a.strip()]
    k_checkpoints = (
        sorted(int(k.strip()) for k in args.k_checkpoints.split(",") if k.strip())
        if args.k_checkpoints
        else sorted(cfg.table_k_checkpoints)
    )

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(__file__).resolve().parent / "results" / cfg.experiment_id
    )

    rows = compute_rows(cfg, arm_prefixes, k_checkpoints, variant_label=args.variant_label)
    write_csv(
        rows,
        out_dir / "per_task_table11.csv",
        fieldnames=["arm", "milestone", "task_idx", "k", "mic"],
    )
    write_csv(
        summarize(rows),
        out_dir / "summary_table11.csv",
        fieldnames=["arm", "milestone", "k", "n_tasks", "mean_mic", "std_mic"],
    )


if __name__ == "__main__":
    main()
