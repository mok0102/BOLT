"""Plot fixed_target_rejection_bo.py's results: N arms' actual downstream BO
objective (best MIC found, lower = better) when every task is initialized
with the *same* fixed-size feasible pool (target_pool_size), across
#tasks trained (milestone). Separate figures per task_set
(trainset / heldout).

Three figures:
- fig1_bo_objective_by_milestone_<task_set>.png: headline chart at one
  reference target_pool_size (default: largest present) and one reference k
  (default 5000 -- additional oracle calls after the fixed init pool).
- fig2_target_sensitivity_<task_set>.png: small multiples, one subplot per
  target_pool_size, to show the trend isn't an artifact of the chosen target.
- fig3_rejection_rate_by_milestone_<task_set>.png: mean rejection rate
  (1 - target_pool_size / draws_used, i.e. how many raw draws it took to
  reach the fixed target) vs. milestone, one line per arm, at the same
  reference target_pool_size fig1 uses (rejection rate is target-dependent).

coverage_rate is left as CSV-only (no dedicated chart), same convention as
the rest of this package.

Usage (run from the BOLT repo root):
    python experiments/eval/plot_fixed_target_rejection_bo.py \\
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
    frames = [pd.read_csv(d / "per_task_fixed_target_bo.csv") for d in results_dirs]
    return pd.concat(frames, ignore_index=True).dropna(subset=["best_mic"])


def plot_fig1(df: pd.DataFrame, task_set: str, out_path: Path, target: int, k: int) -> None:
    sub = df[(df["k"] == k) & (df["task_set"] == task_set) & (df["target_pool_size"] == target)]
    if sub.empty:
        print(f"[plot_fixed_target_rejection_bo] no data for task_set={task_set} target={target} k={k}, skipping {out_path}")
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
        f"BO objective vs. #tasks trained (target pool={target}, k={k}, {TASK_SET_LABEL[task_set]})",
        color="#0b0b0b",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_fig2(df: pd.DataFrame, task_set: str, out_path: Path, k: int) -> None:
    sub = df[(df["task_set"] == task_set) & (df["k"] == k)]
    if sub.empty:
        print(f"[plot_fixed_target_rejection_bo] no data for task_set={task_set} k={k}, skipping {out_path}")
        return
    targets = sorted(sub["target_pool_size"].unique())
    milestones = sorted(sub["milestone"].unique())
    arms = sorted_arms(sub)
    colors = arm_colors(arms)
    n_cols = 3
    n_rows = -(-len(targets) // n_cols)  # ceil div

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharey=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, target in zip(axes, targets):
        panel = sub[sub["target_pool_size"] == target]
        for arm in arms:
            arm_panel = panel[panel["arm"] == arm]
            means = arm_panel.groupby("milestone")["best_mic"].mean().reindex(milestones)
            ax.plot(milestones, means, color=colors[arm], linewidth=2, marker="o", markersize=5)
        ax.set_title(f"target={target}", color="#0b0b0b")
        ax.set_xticks(milestones)
        style_axis(ax)
        ax.tick_params(labelsize=8)

    for ax in axes[len(targets):]:
        ax.axis("off")

    handles = [plt.Line2D([0], [0], color=colors[arm], linewidth=2.5) for arm in arms]
    fig.legend(handles, arms, loc="lower center", ncol=len(arms), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"BO objective: target-pool-size sensitivity check ({TASK_SET_LABEL[task_set]}, k={k})", color="#0b0b0b")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_fig3(df: pd.DataFrame, task_set: str, out_path: Path, target: int) -> None:
    sub = df[(df["task_set"] == task_set) & (df["target_pool_size"] == target)].dropna(subset=["rejection_rate"])
    sub = sub.drop_duplicates(subset=["arm", "milestone", "task_idx"])
    if sub.empty:
        print(f"[plot_fixed_target_rejection_bo] no rejection-rate data for task_set={task_set} "
              f"target={target}, skipping {out_path}")
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
    ax.set_ylabel("rejection rate (1 - target / raw draws)", color=MUTED_TEXT)
    ax.set_xticks(milestones)
    style_axis(ax)
    ax.legend(frameon=False)
    fig.suptitle(
        f"Rejection rate vs. #tasks trained (target pool={target}, {TASK_SET_LABEL[task_set]})", color="#0b0b0b",
    )
    fig.subplots_adjust(left=0.18, top=0.85, bottom=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Comma-separated list of results dirs")
    parser.add_argument("--target", type=int, default=None, help="target_pool_size for fig1 (default: largest present)")
    parser.add_argument("--k", type=int, default=5000, help="k for both figures")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    results_dirs = [Path(d.strip()) for d in args.results_dir.split(",") if d.strip()]
    df = load_per_task(results_dirs)
    plots_dir = resolve_out_dir(results_dirs, args.out_dir) / "plots"
    target = args.target if args.target is not None else (int(df["target_pool_size"].max()) if not df.empty else None)

    for task_set in ("trainset", "heldout"):
        if target is not None:
            plot_fig1(df, task_set, plots_dir / f"fig1_bo_objective_by_milestone_{task_set}.png", target=target, k=args.k)
            plot_fig3(df, task_set, plots_dir / f"fig3_rejection_rate_by_milestone_{task_set}.png", target=target)
        plot_fig2(df, task_set, plots_dir / f"fig2_target_sensitivity_{task_set}.png", k=args.k)


if __name__ == "__main__":
    main()
