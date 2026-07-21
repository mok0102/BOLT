"""Plot the constraint-violation-rate results produced by
measure_violation_rate.py: N arms (e.g. BOLT / ORPT / ORPT-LEX) across
#tasks trained (milestone). Accepts multiple --results-dir values (one per
experiment_id) so a variant trained under its own experiment_id/run_dir --
e.g. ORPT-LEX, tagged via measure_violation_rate.py's --variant-label so it
doesn't collide with another experiment's "ORPT" rows -- can be compared
against another experiment's BOLT/ORPT baseline in the same chart.

Held-out and trained peptides are separate figures (not 1x2 subplots of one
figure) -- with 3+ arms now in every chart, a shared panel got cramped.

Two figure pairs (design rationale in
/root/.claude/plans/orpt-beta-0-25-parsed-turing.md, consulted via the
dataviz skill), one per task_set (heldout20 / trainset):
- fig1_violation_by_milestone_<task_set>.png: headline chart at a fixed k
  (default 100). One line per arm, +/-1 std band across the task set read
  from per_task_violation_rate.csv.
- fig2_k_sensitivity_<task_set>.png: small multiples, one subplot per k,
  one line per arm, to confirm fig1's trend isn't an artifact of the chosen k.

Palette/style/multi-results-dir-merge helpers shared with the other
plot_*.py scripts in this directory live in plot_common.py.

Usage (run from the BOLT repo root):
    python experiments/constraint_violation/plot_violation_rate.py \\
        --results-dir experiments/constraint_violation/results/peptide_100task_orpt_beta0.25

    # combine two experiments' results into one comparison chart:
    python experiments/constraint_violation/plot_violation_rate.py \\
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


def plot_fig1(df: pd.DataFrame, task_set: str, out_path: Path, k: int) -> None:
    sub = df[(df["k"] == k) & (df["task_set"] == task_set)]
    milestones = sorted(sub["milestone"].unique())
    arms = sorted_arms(sub)
    colors = arm_colors(arms)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for arm in arms:
        arm_sub = sub[sub["arm"] == arm]
        stats = arm_sub.groupby("milestone")["violation_rate"].agg(["mean", "std"]).reindex(milestones)
        ax.plot(
            milestones, stats["mean"], color=colors[arm], linewidth=2, marker="o", markersize=8, label=arm,
        )
        ax.fill_between(
            milestones,
            stats["mean"] - stats["std"].fillna(0),
            stats["mean"] + stats["std"].fillna(0),
            color=colors[arm],
            alpha=0.15,
            linewidth=0,
        )
    ax.set_xlabel("#tasks trained (milestone)", color=MUTED_TEXT)
    ax.set_ylabel("constraint violation rate", color=MUTED_TEXT)
    ax.set_xticks(milestones)
    ax.set_ylim(bottom=0)
    style_axis(ax)
    ax.legend(frameon=False)
    fig.suptitle(f"Constraint violation rate vs. #tasks trained (k={k}, {TASK_SET_LABEL[task_set]})", color="#0b0b0b")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_fig2(df: pd.DataFrame, task_set: str, out_path: Path) -> None:
    sub = df[df["task_set"] == task_set]
    ks = sorted(sub["k"].unique())
    milestones = sorted(sub["milestone"].unique())
    arms = sorted_arms(sub)
    colors = arm_colors(arms)
    n_cols = 3
    n_rows = -(-len(ks) // n_cols)  # ceil div

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharey=True)
    axes = axes.flatten()

    for ax, k in zip(axes, ks):
        panel_k = sub[sub["k"] == k]
        for arm in arms:
            panel = panel_k[panel_k["arm"] == arm]
            means = panel.groupby("milestone")["violation_rate"].mean().reindex(milestones)
            ax.plot(milestones, means, color=colors[arm], linewidth=2, marker="o", markersize=5)
        ax.set_title(f"k={k}", color="#0b0b0b")
        ax.set_xticks(milestones)
        ax.set_ylim(bottom=0)
        style_axis(ax)
        ax.tick_params(labelsize=8)

    for ax in axes[len(ks):]:
        ax.axis("off")

    handles = [plt.Line2D([0], [0], color=colors[arm], linewidth=2.5) for arm in arms]
    fig.legend(handles, arms, loc="lower center", ncol=len(arms), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Constraint violation rate: k-sensitivity check ({TASK_SET_LABEL[task_set]})", color="#0b0b0b")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Comma-separated list of results dirs, each containing its own "
        "per_task_violation_rate.csv (one per experiment_id). Rows from all of them "
        "are concatenated before plotting, so arms from different experiments (e.g. "
        "BOLT/ORPT from one, ORPT-LEX from another) appear together.",
    )
    parser.add_argument("--k", type=int, default=100, help="k for the fig1 headline chart")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write plots/ (default: the single --results-dir given, or "
        "results/comparison/ next to this script if multiple were given)",
    )
    args = parser.parse_args()

    results_dirs = [Path(d.strip()) for d in args.results_dir.split(",") if d.strip()]
    df = load_concat_csv(results_dirs, "per_task_violation_rate.csv")
    plots_dir = resolve_out_dir(results_dirs, args.out_dir) / "plots"

    for task_set in ("heldout20", "trainset"):
        plot_fig1(df, task_set, plots_dir / f"fig1_violation_by_milestone_{task_set}.png", k=args.k)
        plot_fig2(df, task_set, plots_dir / f"fig2_k_sensitivity_{task_set}.png")


if __name__ == "__main__":
    main()
