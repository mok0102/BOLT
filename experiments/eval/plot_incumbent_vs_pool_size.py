"""Plot incumbent_vs_pool_size.py's results: N arms' proposal-level
incumbent (best MIC among the first n_proposals feasible proposals, lower =
better) across #tasks trained (milestone). Separate figures per task_set
(trainset / heldout).

Two figure pairs, mirroring the rest of experiments/eval/'s plot scripts:
- incumbent_mic_bymilestone_n<n_proposals>_<task_set>.png: headline chart at
  a fixed n_proposals (default 10). One line per arm, +/-1 std band across
  tasks.
- incumbent_mic_byNProposals_<task_set>.png: small multiples, one subplot
  per n_proposals checkpoint.

rejection_rate_at_n_proposals is left as CSV-only (no dedicated chart).

A third figure, incumbent_coverage_bymilestone_n<n_proposals>_<task_set>.png,
plots coverage_rate_at_n_proposals (fraction of tasks with enough feasible
proposals to even measure an incumbent at n_proposals) vs. milestone -- reads
summary_incumbent_vs_pool_size.csv directly since that rate is already
task-aggregated there.

Usage (run from the BOLT repo root):
    python experiments/eval/plot_incumbent_vs_pool_size.py \\
        --results-dir experiments/eval/results/poc20_four_arm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_common import MUTED_TEXT, arm_colors, filter_arms, load_concat_csv, resolve_out_dir, sorted_arms, style_axis

TASK_SET_LABEL = {"heldout": "held-out peptides (20)", "trainset": "trained peptides", "heldout100": "held-out peptides (100)"}


def plot_fig1(df: pd.DataFrame, task_set: str, out_path: Path, n_proposals: int) -> None:
    sub = df[(df["n_proposals"] == n_proposals) & (df["task_set"] == task_set)].dropna(subset=["incumbent_mic"])
    if sub.empty:
        print(f"[plot_incumbent_vs_pool_size] no data for task_set={task_set} n_proposals={n_proposals}, "
              f"skipping {out_path}")
        return
    milestones = sorted(sub["milestone"].unique())
    arms = sorted_arms(sub)
    colors = arm_colors(arms)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for arm in arms:
        arm_sub = sub[sub["arm"] == arm]
        stats = arm_sub.groupby("milestone")["incumbent_mic"].agg(["mean", "std"]).reindex(milestones)
        ax.plot(milestones, stats["mean"], color=colors[arm], linewidth=2, marker="o", markersize=8, label=arm)
        ax.fill_between(
            milestones,
            stats["mean"] - stats["std"].fillna(0),
            stats["mean"] + stats["std"].fillna(0),
            color=colors[arm],
            alpha=0.15,
            linewidth=0,
        )
    ax.set_xlabel("#tasks trained (milestone)", color=MUTED_TEXT)
    ax.set_ylabel("incumbent MIC (lower = better)", color=MUTED_TEXT)
    ax.set_xticks(milestones)
    style_axis(ax)
    ax.legend(frameon=False)
    fig.suptitle(
        f"Proposal-level incumbent vs. #tasks trained (n_proposals={n_proposals}, {TASK_SET_LABEL[task_set]})",
        color="#0b0b0b",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_fig2(df: pd.DataFrame, task_set: str, out_path: Path) -> None:
    sub = df[df["task_set"] == task_set].dropna(subset=["incumbent_mic"])
    if sub.empty:
        print(f"[plot_incumbent_vs_pool_size] no data for task_set={task_set}, skipping {out_path}")
        return
    n_proposals_values = sorted(sub["n_proposals"].unique())
    milestones = sorted(sub["milestone"].unique())
    arms = sorted_arms(sub)
    colors = arm_colors(arms)
    n_cols = 3
    n_rows = -(-len(n_proposals_values) // n_cols)  # ceil div

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharey=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, n_proposals in zip(axes, n_proposals_values):
        panel_n = sub[sub["n_proposals"] == n_proposals]
        for arm in arms:
            panel = panel_n[panel_n["arm"] == arm]
            means = panel.groupby("milestone")["incumbent_mic"].mean().reindex(milestones)
            ax.plot(milestones, means, color=colors[arm], linewidth=2, marker="o", markersize=5)
        ax.set_title(f"n_proposals={n_proposals}", color="#0b0b0b")
        ax.set_xticks(milestones)
        style_axis(ax)
        ax.tick_params(labelsize=8)

    for ax in axes[len(n_proposals_values):]:
        ax.axis("off")

    for col in range(n_cols):
        rows_in_col = [r for r in range(n_rows) if r * n_cols + col < len(n_proposals_values)]
        if rows_in_col:
            axes[max(rows_in_col) * n_cols + col].set_xlabel("#tasks trained (milestone)", color=MUTED_TEXT)

    handles = [plt.Line2D([0], [0], color=colors[arm], linewidth=2.5) for arm in arms]
    fig.legend(handles, arms, loc="lower center", ncol=len(arms), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Proposal-level incumbent: n_proposals-sensitivity check ({TASK_SET_LABEL[task_set]})", color="#0b0b0b",
    )
    fig.supylabel("incumbent MIC (lower = better)", color=MUTED_TEXT)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_fig3_coverage(df: pd.DataFrame, task_set: str, out_path: Path, n_proposals: int) -> None:
    sub = df[(df["n_proposals"] == n_proposals) & (df["task_set"] == task_set)].dropna(
        subset=["coverage_rate_at_n_proposals"]
    )
    if sub.empty:
        print(f"[plot_incumbent_vs_pool_size] no coverage data for task_set={task_set} n_proposals={n_proposals}, "
              f"skipping {out_path}")
        return
    milestones = sorted(sub["milestone"].unique())
    arms = sorted_arms(sub)
    colors = arm_colors(arms)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for arm in arms:
        arm_sub = sub[sub["arm"] == arm].set_index("milestone").reindex(milestones)
        ax.plot(
            milestones, arm_sub["coverage_rate_at_n_proposals"],
            color=colors[arm], linewidth=2, marker="o", markersize=8, label=arm,
        )
    ax.set_xlabel("#tasks trained (milestone)", color=MUTED_TEXT)
    ax.set_ylabel("coverage rate (fraction of tasks with enough feasible proposals)", color=MUTED_TEXT)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(milestones)
    style_axis(ax)
    ax.legend(frameon=False)
    fig.suptitle(
        f"Coverage rate vs. #tasks trained (n_proposals={n_proposals}, {TASK_SET_LABEL[task_set]})",
        color="#0b0b0b",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Comma-separated list of results dirs")
    parser.add_argument("--n-proposals", type=int, default=10, help="n_proposals for the headline chart")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--arms", default=None, help="Comma-separated subset of arms to plot (default: all present)")
    args = parser.parse_args()

    results_dirs = [Path(d.strip()) for d in args.results_dir.split(",") if d.strip()]
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else None
    df = filter_arms(load_concat_csv(results_dirs, "per_task_incumbent_vs_pool_size.csv"), arms)
    summary_df = filter_arms(load_concat_csv(results_dirs, "summary_incumbent_vs_pool_size.csv"), arms)
    plots_dir = resolve_out_dir(results_dirs, args.out_dir) / "plots"

    # Derived from whatever's actually present in the loaded data, not a
    # hardcoded ("trainset", "heldout") pair -- so results computed with a
    # task_set common.py didn't know about yet at the time this was written
    # (e.g. "heldout100") still get plotted, not silently skipped.
    present_task_sets = sorted(set(df["task_set"]) | set(summary_df["task_set"]))
    for task_set in present_task_sets:
        plot_fig1(
            df, task_set, plots_dir / f"incumbent_mic_bymilestone_n{args.n_proposals}_{task_set}.png",
            n_proposals=args.n_proposals,
        )
        plot_fig2(df, task_set, plots_dir / f"incumbent_mic_byNProposals_{task_set}.png")
        plot_fig3_coverage(
            summary_df, task_set,
            plots_dir / f"incumbent_coverage_bymilestone_n{args.n_proposals}_{task_set}.png",
            n_proposals=args.n_proposals,
        )


if __name__ == "__main__":
    main()
