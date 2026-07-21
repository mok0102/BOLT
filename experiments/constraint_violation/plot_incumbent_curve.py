"""Plot the fine-grained incumbent (best-found-so-far) curves produced by
incumbent_curve.py: a real BO convergence curve -- x = additional oracle
calls (k) after each task's own rejection-sampled init pool, y = mean best
MIC found so far across tasks (+/-1 std band) -- one line per arm, at a
fixed (default: fully-trained) milestone. This is the "final capability"
comparison: how do the arms actually converge once BO starts, not just
their snapshot value at a few sparse k checkpoints
(plot_rejection_sampled_bo.py) or their milestone-by-milestone init-only
quality (plot_table11.py).

Held-out and trained peptides are separate figures, matching every other
plot_*.py in this directory.

x-axis is log-scaled (k spans 3 orders of magnitude, and most of the
interesting movement happens early). Each figure's subtitle lists every
arm's mean init-pool size for this milestone/task_set -- since x=k is
*additional* calls after a variable-size pool (see incumbent_curve.py's
docstring for why total-budget wasn't used as the x-axis instead), an arm
with a much bigger pool (e.g. BOLT, ~900-1400) has actually spent more total
budget than an arm with a small one (e.g. ORPT-LEX, ~5-600) by the same k --
the annotation keeps that from being read as a fully apples-to-apples
budget comparison.

Not built here (per user, not thought through yet): how the BO population's
score *distribution* (not just its best value) evolves over the course of
one run. A natural follow-up once that framing exists -- would likely also
read from these same per-task trajectory CSVs.

Usage (run from the BOLT repo root):
    python experiments/constraint_violation/plot_incumbent_curve.py \\
        --results-dir experiments/constraint_violation/results/peptide_100task_orpt_beta0.25,experiments/constraint_violation/results/peptide_100task_orpt_lexicographic \\
        --out-dir experiments/constraint_violation/results/comparison_beta0.25_vs_lexicographic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_common import MUTED_TEXT, arm_colors, load_concat_csv, resolve_out_dir, sorted_arms, style_axis

TASK_SET_LABEL = {"heldout20": "held-out peptides", "trainset": "trained peptides"}


def load_summary(results_dirs: list[Path]) -> pd.DataFrame:
    return load_concat_csv(results_dirs, "summary_incumbent_curve.csv")


def plot_incumbent_curve(df: pd.DataFrame, task_set: str, milestone: int, out_path: Path) -> None:
    sub = df[(df["task_set"] == task_set) & (df["milestone"] == milestone)]
    if sub.empty:
        print(f"[plot_incumbent_curve] no rows for task_set={task_set}, milestone={milestone}, skipping")
        return
    arms = sorted_arms(sub)
    colors = arm_colors(arms)

    fig, ax = plt.subplots(figsize=(7, 5))
    pool_size_labels = []
    for arm in arms:
        arm_sub = sub[sub["arm"] == arm].sort_values("k")
        ax.plot(
            arm_sub["k"], arm_sub["mean_best_mic"], color=colors[arm], linewidth=2, label=arm,
        )
        ax.fill_between(
            arm_sub["k"],
            arm_sub["mean_best_mic"] - arm_sub["std_best_mic"].fillna(0),
            arm_sub["mean_best_mic"] + arm_sub["std_best_mic"].fillna(0),
            color=colors[arm],
            alpha=0.15,
            linewidth=0,
        )
        pool_size_labels.append(f"{arm}={arm_sub['mean_pool_size'].mean():.0f}")

    ax.set_xscale("log")
    ax.set_xlabel("additional oracle calls after init (k)", color=MUTED_TEXT)
    ax.set_ylabel("mean best MIC found so far (lower = better)", color=MUTED_TEXT)
    style_axis(ax)
    ax.legend(frameon=False)
    fig.suptitle(f"Incumbent performance vs. oracle calls (milestone={milestone}, {TASK_SET_LABEL[task_set]})",
                 color="#0b0b0b")
    ax.set_title(f"mean init pool size: {', '.join(pool_size_labels)}", color=MUTED_TEXT, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Comma-separated list of results dirs, each containing its own "
        "summary_incumbent_curve.csv (one per experiment_id).",
    )
    parser.add_argument(
        "--milestone",
        type=int,
        default=None,
        help="Which milestone to plot (default: the largest one present, i.e. the fully-trained "
        "checkpoint)",
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    results_dirs = [Path(d.strip()) for d in args.results_dir.split(",") if d.strip()]
    df = load_summary(results_dirs)
    milestone = args.milestone if args.milestone is not None else int(df["milestone"].max())
    plots_dir = resolve_out_dir(results_dirs, args.out_dir) / "plots"

    for task_set in ("heldout20", "trainset"):
        plot_incumbent_curve(df, task_set, milestone, plots_dir / f"fig_incumbent_by_oracle_calls_{task_set}.png")


if __name__ == "__main__":
    main()
