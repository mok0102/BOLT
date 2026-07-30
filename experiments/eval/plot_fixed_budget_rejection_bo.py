"""Plot fixed_budget_rejection_bo.py's results: N arms' actual downstream BO
objective (best MIC found, lower = better) under a real, variable-size
reject-and-discard init pool (whatever a fixed sampling budget yields),
across #tasks trained (milestone). Separate figures per task_set
(trainset / heldout).

Three figures:
- fixedbudget_mic_bymilestone_bo<bo_calls>_<task_set>.png: headline chart at
  a fixed bo_calls (default 5000 -- additional oracle calls made by the BO
  loop after the init pool; 5000 happens to equal the full oracle budget,
  i.e. the "final" BO outcome).
- fixedbudget_mic_byboCalls_<task_set>.png: small multiples, one subplot per
  bo_calls checkpoint, to show the trend isn't an artifact of the chosen
  checkpoint.
- fixedbudget_rejection_bymilestone_<task_set>.png: mean rejection rate
  (1 - pool_size / draws_used, i.e. how much of the raw sampling budget a
  reject-and-discard policy had to throw away to build the pool it did get)
  vs. milestone, one line per arm. rejection_rate is per-task/per-milestone,
  not per-bo_calls, so this dedupes per_task rows to one per task before
  aggregating.

A fourth figure, fixedbudget_coverage_bymilestone_<task_set>.png, plots
coverage_rate (fraction of tasks clearing the min_feasible floor) vs.
milestone -- reads fixed_budget_bo_coverage.csv directly since that rate is
already task-aggregated there. Pool-size spread is left as CSV-only.

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


def plot_fig1(df: pd.DataFrame, task_set: str, out_path: Path, bo_calls: int) -> None:
    sub = df[(df["bo_calls"] == bo_calls) & (df["task_set"] == task_set)]
    if sub.empty:
        print(f"[plot_fixed_budget_rejection_bo] no data for task_set={task_set} bo_calls={bo_calls}, "
              f"skipping {out_path}")
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
    fig.suptitle(
        f"BO objective vs. #tasks trained ({bo_calls} BO calls after init pool, {TASK_SET_LABEL[task_set]})",
        color="#0b0b0b",
    )
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
    bo_calls_values = sorted(sub["bo_calls"].unique())
    milestones = sorted(sub["milestone"].unique())
    arms = sorted_arms(sub)
    colors = arm_colors(arms)
    n_cols = 3
    n_rows = -(-len(bo_calls_values) // n_cols)  # ceil div

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharey=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, bo_calls in zip(axes, bo_calls_values):
        panel_bo_calls = sub[sub["bo_calls"] == bo_calls]
        for arm in arms:
            panel = panel_bo_calls[panel_bo_calls["arm"] == arm]
            means = panel.groupby("milestone")["best_mic"].mean().reindex(milestones)
            ax.plot(milestones, means, color=colors[arm], linewidth=2, marker="o", markersize=5)
        ax.set_title(f"bo_calls={bo_calls}", color="#0b0b0b")
        ax.set_xticks(milestones)
        style_axis(ax)
        ax.tick_params(labelsize=8)

    for ax in axes[len(bo_calls_values):]:
        ax.axis("off")

    for col in range(n_cols):
        rows_in_col = [r for r in range(n_rows) if r * n_cols + col < len(bo_calls_values)]
        if rows_in_col:
            axes[max(rows_in_col) * n_cols + col].set_xlabel("#tasks trained (milestone)", color=MUTED_TEXT)

    handles = [plt.Line2D([0], [0], color=colors[arm], linewidth=2.5) for arm in arms]
    fig.legend(handles, arms, loc="lower center", ncol=len(arms), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"BO objective (best MIC found): bo_calls-sensitivity check ({TASK_SET_LABEL[task_set]})", color="#0b0b0b",
    )
    fig.supylabel("best MIC found (lower = more potent)", color=MUTED_TEXT)
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
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def load_coverage(results_dirs: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(d / "fixed_budget_bo_coverage.csv") for d in results_dirs]
    return pd.concat(frames, ignore_index=True)


def plot_fig4_coverage(df: pd.DataFrame, task_set: str, out_path: Path) -> None:
    sub = df[df["task_set"] == task_set].dropna(subset=["coverage_rate"])
    if sub.empty:
        print(f"[plot_fixed_budget_rejection_bo] no coverage data for task_set={task_set}, skipping {out_path}")
        return
    milestones = sorted(sub["milestone"].unique())
    arms = sorted_arms(sub)
    colors = arm_colors(arms)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for arm in arms:
        arm_sub = sub[sub["arm"] == arm].set_index("milestone").reindex(milestones)
        ax.plot(
            milestones, arm_sub["coverage_rate"],
            color=colors[arm], linewidth=2, marker="o", markersize=8, label=arm,
        )
    ax.set_xlabel("#tasks trained (milestone)", color=MUTED_TEXT)
    ax.set_ylabel("coverage rate (fraction of tasks clearing min_feasible)", color=MUTED_TEXT)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(milestones)
    style_axis(ax)
    ax.legend(frameon=False)
    fig.suptitle(f"Coverage rate vs. #tasks trained (fixed budget, {TASK_SET_LABEL[task_set]})", color="#0b0b0b")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Comma-separated list of results dirs")
    parser.add_argument("--bo-calls", type=int, default=5000, help="bo_calls for the headline chart")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    results_dirs = [Path(d.strip()) for d in args.results_dir.split(",") if d.strip()]
    df = load_per_task(results_dirs)
    coverage_df = load_coverage(results_dirs)
    plots_dir = resolve_out_dir(results_dirs, args.out_dir) / "plots"

    for task_set in ("trainset", "heldout"):
        plot_fig1(
            df, task_set, plots_dir / f"fixedbudget_mic_bymilestone_bo{args.bo_calls}_{task_set}.png",
            bo_calls=args.bo_calls,
        )
        plot_fig2(df, task_set, plots_dir / f"fixedbudget_mic_byboCalls_{task_set}.png")
        plot_fig3(df, task_set, plots_dir / f"fixedbudget_rejection_bymilestone_{task_set}.png")
        plot_fig4_coverage(coverage_df, task_set, plots_dir / f"fixedbudget_coverage_bymilestone_{task_set}.png")


if __name__ == "__main__":
    main()
