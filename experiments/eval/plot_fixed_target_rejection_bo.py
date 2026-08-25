"""Plot fixed_target_rejection_bo.py's results: N arms' actual downstream BO
objective (best MIC found, lower = better) when every task is initialized
with the *same* fixed-size feasible pool (target_pool_size), across
#tasks trained (milestone). Separate figures per task_set
(trainset / heldout).

Three figures:
- fixedtarget_mic_bymilestone_target<T>_bo<bo_calls>_<task_set>.png: headline
  chart at one reference target_pool_size (default: largest present) and one
  reference bo_calls (default 5000 -- additional oracle calls after the
  fixed init pool).
- fixedtarget_mic_bytarget_bo<bo_calls>_<task_set>.png: small multiples, one
  subplot per target_pool_size, to show the trend isn't an artifact of the
  chosen target.
- fixedtarget_rejection_bymilestone_target<T>_<task_set>.png: mean rejection
  rate (1 - target_pool_size / draws_used, i.e. how many raw draws it took
  to reach the fixed target) vs. milestone, one line per arm, at the same
  reference target_pool_size the headline chart uses (rejection rate is
  target-dependent, not bo_calls-dependent).

A fourth figure, fixedtarget_coverage_bymilestone_target<T>_<task_set>.png,
plots coverage_rate (fraction of tasks that could reach target_pool_size at
all) vs. milestone at the same reference target the headline chart uses --
reads fixed_target_bo_coverage.csv directly since that rate is already
task-aggregated there.

Usage (run from the BOLT repo root):
    python experiments/eval/plot_fixed_target_rejection_bo.py \\
        --results-dir experiments/eval/results/poc20_four_arm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_common import (
    MUTED_TEXT,
    arm_colors,
    arm_legend_handles,
    filter_arms,
    is_milestone_independent,
    plot_arm_line,
    resolve_out_dir,
    sorted_arms,
    style_axis,
)

TASK_SET_LABEL = {"heldout": "held-out peptides", "trainset": "trained peptides"}


def load_per_task(results_dirs: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(d / "per_task_fixed_target_bo.csv") for d in results_dirs]
    return pd.concat(frames, ignore_index=True).dropna(subset=["best_mic"])


def plot_fig1(df: pd.DataFrame, task_set: str, out_path: Path, target: int, bo_calls: int) -> None:
    sub = df[(df["bo_calls"] == bo_calls) & (df["task_set"] == task_set) & (df["target_pool_size"] == target)]
    if sub.empty:
        print(f"[plot_fixed_target_rejection_bo] no data for task_set={task_set} target={target} "
              f"bo_calls={bo_calls}, skipping {out_path}")
        return
    milestones = sorted(sub["milestone"].unique())
    arms = sorted_arms(sub)
    colors = arm_colors(arms)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for arm in arms:
        arm_sub = sub[sub["arm"] == arm]
        stats = arm_sub.groupby("milestone")["best_mic"].agg(["mean", "std"]).reindex(milestones)
        plot_arm_line(ax, milestones, stats["mean"], colors[arm], stds=stats["std"], label=arm)
    ax.set_xlabel("#tasks trained (milestone)", color=MUTED_TEXT)
    ax.set_ylabel("best MIC found (lower = more potent)", color=MUTED_TEXT)
    ax.set_xticks(milestones)
    style_axis(ax)
    ax.legend(frameon=False)
    fig.suptitle(
        f"BO objective vs. #tasks trained (target pool={target}, bo_calls={bo_calls}, {TASK_SET_LABEL[task_set]})",
        color="#0b0b0b",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_fig2(df: pd.DataFrame, task_set: str, out_path: Path, bo_calls: int) -> None:
    sub = df[(df["task_set"] == task_set) & (df["bo_calls"] == bo_calls)]
    if sub.empty:
        print(f"[plot_fixed_target_rejection_bo] no data for task_set={task_set} bo_calls={bo_calls}, "
              f"skipping {out_path}")
        return
    targets = sorted(sub["target_pool_size"].unique())
    milestones = sorted(sub["milestone"].unique())
    arms = sorted_arms(sub)
    colors = arm_colors(arms)
    n_cols = 3
    n_rows = -(-len(targets) // n_cols)  # ceil div

    flat_arms = {arm for arm in arms if is_milestone_independent(sub[sub["arm"] == arm])}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharey=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, target in zip(axes, targets):
        panel = sub[sub["target_pool_size"] == target]
        for arm in arms:
            arm_panel = panel[panel["arm"] == arm]
            means = arm_panel.groupby("milestone")["best_mic"].mean().reindex(milestones)
            plot_arm_line(ax, milestones, means, colors[arm], marker_size=5)
        ax.set_title(f"target={target}", color="#0b0b0b")
        ax.set_xticks(milestones)
        style_axis(ax)
        ax.tick_params(labelsize=8)

    for ax in axes[len(targets):]:
        ax.axis("off")

    for col in range(n_cols):
        rows_in_col = [r for r in range(n_rows) if r * n_cols + col < len(targets)]
        if rows_in_col:
            axes[max(rows_in_col) * n_cols + col].set_xlabel("#tasks trained (milestone)", color=MUTED_TEXT)

    handles = arm_legend_handles(arms, colors, flat_arms)
    fig.legend(handles, arms, loc="lower center", ncol=len(arms), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"BO objective: target-pool-size sensitivity check ({TASK_SET_LABEL[task_set]}, bo_calls={bo_calls})",
        color="#0b0b0b",
    )
    fig.supylabel("best MIC found (lower = more potent)", color=MUTED_TEXT)
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
        plot_arm_line(ax, milestones, stats["mean"], colors[arm], stds=stats["std"], label=arm)
    ax.set_xlabel("#tasks trained (milestone)", color=MUTED_TEXT)
    ax.set_ylabel("rejection rate (1 - target / raw draws)", color=MUTED_TEXT)
    ax.set_xticks(milestones)
    style_axis(ax)
    ax.legend(frameon=False)
    fig.suptitle(
        f"Rejection rate vs. #tasks trained (target pool={target}, {TASK_SET_LABEL[task_set]})", color="#0b0b0b",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def load_coverage(results_dirs: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(d / "fixed_target_bo_coverage.csv") for d in results_dirs]
    return pd.concat(frames, ignore_index=True)


def plot_fig4_coverage(df: pd.DataFrame, task_set: str, out_path: Path, target: int) -> None:
    sub = df[(df["task_set"] == task_set) & (df["target_pool_size"] == target)].dropna(subset=["coverage_rate"])
    if sub.empty:
        print(f"[plot_fixed_target_rejection_bo] no coverage data for task_set={task_set} target={target}, "
              f"skipping {out_path}")
        return
    milestones = sorted(sub["milestone"].unique())
    arms = sorted_arms(sub)
    colors = arm_colors(arms)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for arm in arms:
        arm_sub = sub[sub["arm"] == arm].set_index("milestone").reindex(milestones)
        plot_arm_line(ax, milestones, arm_sub["coverage_rate"], colors[arm], label=arm)
    ax.set_xlabel("#tasks trained (milestone)", color=MUTED_TEXT)
    ax.set_ylabel("coverage rate (fraction of tasks reaching target pool size)", color=MUTED_TEXT)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(milestones)
    style_axis(ax)
    ax.legend(frameon=False)
    fig.suptitle(
        f"Coverage rate vs. #tasks trained (target pool={target}, {TASK_SET_LABEL[task_set]})", color="#0b0b0b",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Comma-separated list of results dirs")
    parser.add_argument(
        "--target", type=int, default=None, help="target_pool_size for the headline chart (default: largest present)"
    )
    parser.add_argument("--bo-calls", type=int, default=5000, help="bo_calls for both figures")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--arms", default=None, help="Comma-separated subset of arms to plot (default: all present)")
    args = parser.parse_args()

    results_dirs = [Path(d.strip()) for d in args.results_dir.split(",") if d.strip()]
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else None
    df = filter_arms(load_per_task(results_dirs), arms)
    coverage_df = filter_arms(load_coverage(results_dirs), arms)
    plots_dir = resolve_out_dir(results_dirs, args.out_dir) / "plots"
    target = args.target if args.target is not None else (int(df["target_pool_size"].max()) if not df.empty else None)

    for task_set in ("trainset", "heldout"):
        if target is not None:
            plot_fig1(
                df, task_set,
                plots_dir / f"fixedtarget_mic_bymilestone_target{target}_bo{args.bo_calls}_{task_set}.png",
                target=target, bo_calls=args.bo_calls,
            )
            plot_fig3(
                df, task_set, plots_dir / f"fixedtarget_rejection_bymilestone_target{target}_{task_set}.png",
                target=target,
            )
            plot_fig4_coverage(
                coverage_df, task_set,
                plots_dir / f"fixedtarget_coverage_bymilestone_target{target}_{task_set}.png",
                target=target,
            )
        plot_fig2(
            df, task_set, plots_dir / f"fixedtarget_mic_bytarget_bo{args.bo_calls}_{task_set}.png",
            bo_calls=args.bo_calls,
        )


if __name__ == "__main__":
    main()
