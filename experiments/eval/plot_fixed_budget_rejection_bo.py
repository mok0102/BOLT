"""Plot fixed_budget_rejection_bo.py's results: N arms' actual downstream BO
objective (best MIC found, lower = better) under a real, variable-size
reject-and-discard init pool (whatever a fixed sampling budget yields),
across #tasks trained (milestone). Separate figures per task_set
(trainset / heldout).

Three figures:
- fig1_bo_objective_by_milestone_<task_set>.png: headline chart at a fixed k
  (default 5000 = the full oracle budget, i.e. the "final" BO outcome).
- fig2_bo_objective_k_sensitivity_<task_set>.png: small multiples, one
  subplot per k, to show the trend isn't an artifact of the chosen k.
- fig3_rejection_rate_by_milestone_<task_set>.png: mean rejection rate
  (1 - pool_size / draws_used, i.e. how much of the raw sampling budget a
  reject-and-discard policy had to throw away to build the pool it did get)
  vs. milestone, one line per arm. rejection_rate is per-task/per-milestone,
  not per-k, so this dedupes per_task rows to one per task before
  aggregating.

coverage_rate / pool-size spread are left as CSV-only (no dedicated chart).

Usage (run from the BOLT repo root):
    python experiments/eval/plot_fixed_budget_rejection_bo.py \\
        --results-dir experiments/eval/results/poc20_four_arm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_common import MUTED_TEXT, arm_colors, resolve_out_dir, sorted_arms, style_axis

TASK_SET_LABEL = {"heldout": "held-out peptides", "trainset": "trained peptides"}


def load_per_task(results_dirs: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(d / "per_task_fixed_budget_bo.csv") for d in results_dirs]
    return pd.concat(frames, ignore_index=True).dropna(subset=["best_mic"])


def plot_fig1(df: pd.DataFrame, task_set: str, out_path: Path, k: int) -> None:
    sub = df[(df["k"] == k) & (df["task_set"] == task_set)]
    if sub.empty:
        print(f"[plot_fixed_budget_rejection_bo] no data for task_set={task_set} k={k}, skipping {out_path}")
        return
    milestones = sorted(sub["milestone"].unique())
    arms = sorted_arms(sub)
    colors = arm_colors(arms)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for arm in arms:
        arm_sub = sub[sub["arm"] == arm]
        stats = arm_sub.groupby("milestone")["best_mic"].agg(["mean", "std"]).reindex(milestones)
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
    ax.set_ylabel("best MIC found (lower = more potent)", color=MUTED_TEXT)
    ax.set_xticks(milestones)
    style_axis(ax)
    ax.legend(frameon=False)
    fig.suptitle(f"BO objective vs. #tasks trained (k={k} oracle calls, {TASK_SET_LABEL[task_set]})", color="#0b0b0b")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_fig2(df: pd.DataFrame, task_set: str, out_path: Path) -> None:
    sub = df[df["task_set"] == task_set]
    if sub.empty:
        print(f"[plot_fixed_budget_rejection_bo] no data for task_set={task_set}, skipping {out_path}")
        return
    ks = sorted(sub["k"].unique())
    milestones = sorted(sub["milestone"].unique())
    arms = sorted_arms(sub)
    colors = arm_colors(arms)
    n_cols = 3
    n_rows = -(-len(ks) // n_cols)  # ceil div

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharey=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, k in zip(axes, ks):
        panel_k = sub[sub["k"] == k]
        for arm in arms:
            panel = panel_k[panel_k["arm"] == arm]
            means = panel.groupby("milestone")["best_mic"].mean().reindex(milestones)
            ax.plot(milestones, means, color=colors[arm], linewidth=2, marker="o", markersize=5)
        ax.set_title(f"k={k}", color="#0b0b0b")
        ax.set_xticks(milestones)
        style_axis(ax)
        ax.tick_params(labelsize=8)

    for ax in axes[len(ks):]:
        ax.axis("off")

    handles = [plt.Line2D([0], [0], color=colors[arm], linewidth=2.5) for arm in arms]
    fig.legend(handles, arms, loc="lower center", ncol=len(arms), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"BO objective (best MIC found): k-sensitivity check ({TASK_SET_LABEL[task_set]})", color="#0b0b0b")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_fig3(df: pd.DataFrame, task_set: str, out_path: Path) -> None:
    sub = df[df["task_set"] == task_set].dropna(subset=["rejection_rate"])
    sub = sub.drop_duplicates(subset=["arm", "milestone", "task_idx"])
    if sub.empty:
        print(f"[plot_fixed_budget_rejection_bo] no rejection-rate data for task_set={task_set}, skipping {out_path}")
        return
    milestones = sorted(sub["milestone"].unique())
    arms = sorted_arms(sub)
    colors = arm_colors(arms)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for arm in arms:
        arm_sub = sub[sub["arm"] == arm]
        stats = arm_sub.groupby("milestone")["rejection_rate"].agg(["mean", "std"]).reindex(milestones)
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
    ax.set_ylabel("rejection rate (1 - pool size / raw draws)", color=MUTED_TEXT)
    ax.set_xticks(milestones)
    style_axis(ax)
    ax.legend(frameon=False)
    fig.suptitle(f"Rejection rate vs. #tasks trained (fixed budget, {TASK_SET_LABEL[task_set]})", color="#0b0b0b")
    fig.subplots_adjust(left=0.18, top=0.85, bottom=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Comma-separated list of results dirs")
    parser.add_argument("--k", type=int, default=5000, help="k for the fig1 headline chart")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    results_dirs = [Path(d.strip()) for d in args.results_dir.split(",") if d.strip()]
    df = load_per_task(results_dirs)
    plots_dir = resolve_out_dir(results_dirs, args.out_dir) / "plots"

    for task_set in ("trainset", "heldout"):
        plot_fig1(df, task_set, plots_dir / f"fig1_bo_objective_by_milestone_{task_set}.png", k=args.k)
        plot_fig2(df, task_set, plots_dir / f"fig2_bo_objective_k_sensitivity_{task_set}.png")
        plot_fig3(df, task_set, plots_dir / f"fig3_rejection_rate_by_milestone_{task_set}.png")


if __name__ == "__main__":
    main()
