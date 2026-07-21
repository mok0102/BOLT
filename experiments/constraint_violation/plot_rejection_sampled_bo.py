"""Plot the rejection-sampled-BO results produced by run_rejection_sampled_bo.py
+ summarize_rejection_sampled_bo.py: N arms' actual downstream BO
objective (best MIC found, lower = more potent = better), across #tasks
trained (milestone).

Held-out and trained peptides are separate figures (not 1x2 subplots of one
figure) -- with 3+ arms now in every chart, a shared panel got cramped.

Palette/style/multi-results-dir-merge helpers shared with the other
plot_*.py scripts in this directory live in plot_common.py.

Two figure pairs, one per task_set (heldout20 / trainset):
- fig1_bo_objective_by_milestone_<task_set>.png: headline chart at a fixed k
  (default 5000 = the full oracle budget, i.e. the "final" BO outcome). One
  line per arm, +/-1 std band across tasks read from
  per_task_rejection_sampled_bo.csv.
- fig2_bo_objective_k_sensitivity_<task_set>.png: small multiples, one
  subplot per k, one line per arm, to show the trend isn't an artifact of
  the chosen k.

Note on reading these charts: init pool size is NOT fixed across
arms/milestones/tasks by design (see run_rejection_sampled_bo.py's
docstring) -- k is additional oracle calls made *after* whatever pool size
that task actually got, so e.g. BOLT's much larger typical pools (900-1400)
mean small k values mostly just re-report the init pool's own best value
until k is large enough to add real new acquisition steps. k=5000 (the full
budget) is the fairest cross-arm comparison for that reason. See
incumbent_curve.py/plot_incumbent_curve.py for a proper convergence curve
(x=k at fine granularity) rather than milestone-indexed snapshots at a few k.

Usage (run from the BOLT repo root):
    python experiments/constraint_violation/plot_rejection_sampled_bo.py \\
        --results-dir experiments/constraint_violation/results/peptide_100task_orpt_beta0.25

    # combine two experiments' results into one comparison chart:
    python experiments/constraint_violation/plot_rejection_sampled_bo.py \\
        --results-dir experiments/constraint_violation/results/peptide_100task_orpt_beta0.25,experiments/constraint_violation/results/peptide_100task_orpt_lexicographic \\
        --out-dir experiments/constraint_violation/results/comparison_beta0.25_vs_lexicographic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_common import MUTED_TEXT, arm_colors, resolve_out_dir, sorted_arms, style_axis

TASK_SET_LABEL = {"heldout20": "held-out peptides", "trainset": "trained peptides"}


def load_per_task(results_dirs: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(d / "per_task_rejection_sampled_bo.csv") for d in results_dirs]
    return pd.concat(frames, ignore_index=True).dropna(subset=["best_mic"])


def plot_fig1(df: pd.DataFrame, task_set: str, out_path: Path, k: int) -> None:
    sub = df[(df["k"] == k) & (df["task_set"] == task_set)]
    milestones = sorted(sub["milestone"].unique())
    arms = sorted_arms(sub)
    colors = arm_colors(arms)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for arm in arms:
        arm_sub = sub[sub["arm"] == arm]
        stats = arm_sub.groupby("milestone")["best_mic"].agg(["mean", "std"]).reindex(milestones)
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
    ax.set_ylabel("best MIC found (lower = more potent)", color=MUTED_TEXT)
    ax.set_xticks(milestones)
    ax.set_ylim(bottom=0)
    style_axis(ax)
    ax.legend(frameon=False)
    fig.suptitle(f"BO objective vs. #tasks trained (k={k} oracle calls, {TASK_SET_LABEL[task_set]})", color="#0b0b0b")
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
            means = panel.groupby("milestone")["best_mic"].mean().reindex(milestones)
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
    fig.suptitle(f"BO objective (best MIC found): k-sensitivity check ({TASK_SET_LABEL[task_set]})", color="#0b0b0b")
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
        "per_task_rejection_sampled_bo.csv (one per experiment_id).",
    )
    parser.add_argument("--k", type=int, default=5000, help="k for the fig1 headline chart")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    results_dirs = [Path(d.strip()) for d in args.results_dir.split(",") if d.strip()]
    df = load_per_task(results_dirs)
    plots_dir = resolve_out_dir(results_dirs, args.out_dir) / "plots"

    for task_set in ("heldout20", "trainset"):
        plot_fig1(df, task_set, plots_dir / f"fig1_bo_objective_by_milestone_{task_set}.png", k=args.k)
        plot_fig2(df, task_set, plots_dir / f"fig2_bo_objective_k_sensitivity_{task_set}.png")


if __name__ == "__main__":
    main()
