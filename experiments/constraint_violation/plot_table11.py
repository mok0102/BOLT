"""Plot the Table-11-style ("few-shot"/init-only) results produced by
table11_variant.py: N arms' summed-then-averaged MIC (lower = more potent =
better) across #tasks trained (milestone), on the 20-task held-out set.
Unlike violation_rate/rejection_sampled_bo there is only one task_set here
(Table 11 is a held-out-only metric in the paper -- no trained-peptide
side), so each figure is already a single panel.

Mirrors plot_violation_rate.py's structure/palette (shared with the other
plot_*.py scripts via plot_common.py) so all charts in this directory read
as one visual system:
- fig1_table11_by_milestone.png: headline chart at a fixed k (default 100).
  One line per arm, +/-1 std band across the 20 held-out tasks.
- fig2_table11_k_sensitivity.png: small multiples, one subplot per k.

Usage (run from the BOLT repo root):
    python experiments/constraint_violation/plot_table11.py \\
        --results-dir experiments/constraint_violation/results/peptide_100task_orpt_beta0.25,experiments/constraint_violation/results/peptide_100task_orpt_lexicographic \\
        --out-dir experiments/constraint_violation/results/comparison_beta0.25_vs_lexicographic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_common import MUTED_TEXT, arm_colors, load_concat_csv, resolve_out_dir, sorted_arms, style_axis


def plot_fig1(df: pd.DataFrame, out_path: Path, k: int) -> None:
    sub = df[df["k"] == k]
    milestones = sorted(sub["milestone"].unique())
    arms = sorted_arms(sub)
    colors = arm_colors(arms)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for arm in arms:
        arm_sub = sub[sub["arm"] == arm]
        stats = arm_sub.groupby("milestone")["mic"].agg(["mean", "std"]).reindex(milestones)
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
    ax.set_ylabel("mean MIC per task (lower = better)", color=MUTED_TEXT)
    ax.set_xticks(milestones)
    style_axis(ax)
    ax.legend(frameon=False)
    fig.suptitle(f"Table 11 (init-only) MIC vs. #tasks trained (k={k})", color="#0b0b0b")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_fig2(df: pd.DataFrame, out_path: Path) -> None:
    ks = sorted(df["k"].unique())
    milestones = sorted(df["milestone"].unique())
    arms = sorted_arms(df)
    colors = arm_colors(arms)
    n_cols = 3
    n_rows = -(-len(ks) // n_cols)  # ceil div

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharey=True)
    axes = axes.flatten()

    for ax, k in zip(axes, ks):
        panel_k = df[df["k"] == k]
        for arm in arms:
            means = panel_k[panel_k["arm"] == arm].groupby("milestone")["mic"].mean().reindex(milestones)
            ax.plot(milestones, means, color=colors[arm], linewidth=2, marker="o", markersize=5)
        ax.set_title(f"k={k}", color="#0b0b0b")
        ax.set_xticks(milestones)
        style_axis(ax)
        ax.tick_params(labelsize=8)

    for ax in axes[len(ks):]:
        ax.axis("off")

    handles = [plt.Line2D([0], [0], color=colors[arm], linewidth=2.5) for arm in arms]
    fig.legend(handles, arms, loc="lower center", ncol=len(arms), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Table 11 (init-only) MIC: k-sensitivity check", color="#0b0b0b")
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
        help="Comma-separated list of results dirs, each containing its own per_task_table11.csv",
    )
    parser.add_argument("--k", type=int, default=100, help="k for the fig1 headline chart")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    results_dirs = [Path(d.strip()) for d in args.results_dir.split(",") if d.strip()]
    df = load_concat_csv(results_dirs, "per_task_table11.csv")
    plots_dir = resolve_out_dir(results_dirs, args.out_dir) / "plots"

    plot_fig1(df, plots_dir / "fig1_table11_by_milestone.png", k=args.k)
    plot_fig2(df, plots_dir / "fig2_table11_k_sensitivity.png")


if __name__ == "__main__":
    main()
