"""Per-peptide alignment: does a peptide's (whole-pool) feasible frequency
track the best APEX score the model found for it? x-axis = peptides sorted
by that arm's own feasible_frequency (descending); left y-axis = feasible
frequency (bar); right y-axis = best APEX score (line).

This is a genuine dual-axis chart by explicit request -- frequency (0-1) and
unbounded score sharing one panel is normally an anti-pattern (see
plot_rank_alignment.py's stacked-panel design for the alternative), but here
it's what was actually asked for, so it's built as asked rather than
substituted.

Produces three figures from rank_alignment.py's per_peptide_alignment.csv:
- fig_peptide_alignment_BOLT.png / _ORPT.png: one arm at a time, x-tick
  labels are the real task_idx in that arm's own sorted order (rotated 90
  degrees -- meaningful here since only one arm's peptides are on the axis).
- fig_peptide_alignment_overlay.png: BOLT and ORPT overlaid on one panel;
  x-axis is a generic sorted position (1, 2, 3, ...) since the two arms are
  sorted independently and the peptide at a given position differs between
  them -- no single task_idx label would be correct for both bars at that
  position.

Usage (run from the BOLT repo root):
    python experiments/constraint_violation/plot_peptide_alignment.py \\
        --results-dir experiments/constraint_violation/results/peptide_100task_orpt_beta0.25 \\
        --milestone 100 --task-set heldout20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ARM_COLOR = {"BOLT": "#2a78d6", "ORPT": "#e34948"}
GRID_COLOR = "#e1e0d9"
MUTED_TEXT = "#898781"


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.tick_params(colors=MUTED_TEXT)


def plot_single_arm(df: pd.DataFrame, arm: str, milestone: int, task_set: str, out_path: Path) -> None:
    sub = df[(df["arm"] == arm) & (df["milestone"] == milestone) & (df["task_set"] == task_set)]
    sub = sub.sort_values("feasible_frequency", ascending=False).reset_index(drop=True)
    x = range(1, len(sub) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(x, sub["feasible_frequency"], color=ARM_COLOR[arm], alpha=0.55)
    ax1.set_ylabel("feasible frequency", color=MUTED_TEXT)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(sub["task_idx"], rotation=90, fontsize=8)
    ax1.set_xlabel("peptide (task_idx), sorted by feasible frequency", color=MUTED_TEXT)
    ax1.grid(True, axis="y", color=GRID_COLOR, linewidth=1)
    _style_axis(ax1)

    ax2 = ax1.twinx()
    ax2.plot(x, sub["best_score"], color=ARM_COLOR[arm], linewidth=2, marker="o", markersize=4)
    ax2.set_ylabel("best APEX score", color=MUTED_TEXT)
    _style_axis(ax2)

    fig.suptitle(
        f"{arm} (milestone={milestone}, {task_set}): feasible frequency vs. best score per peptide",
        color="#0b0b0b",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_overlay(df: pd.DataFrame, milestone: int, task_set: str, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    width = 0.38
    offsets = {"BOLT": -width / 2, "ORPT": width / 2}

    bar_handles = []
    for arm in ("BOLT", "ORPT"):
        sub = df[(df["arm"] == arm) & (df["milestone"] == milestone) & (df["task_set"] == task_set)]
        sub = sub.sort_values("feasible_frequency", ascending=False).reset_index(drop=True)
        x = range(1, len(sub) + 1)
        bars = ax1.bar(
            [xi + offsets[arm] for xi in x],
            sub["feasible_frequency"],
            width=width,
            color=ARM_COLOR[arm],
            alpha=0.55,
            label=arm,
        )
        bar_handles.append(bars)
        ax2.plot(x, sub["best_score"], color=ARM_COLOR[arm], linewidth=2, marker="o", markersize=4)

    ax1.set_ylabel("feasible frequency", color=MUTED_TEXT)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel(
        "peptide, sorted independently per arm by feasible frequency (generic position -- "
        "underlying peptide differs between BOLT and ORPT at the same position)",
        color=MUTED_TEXT,
    )
    ax1.grid(True, axis="y", color=GRID_COLOR, linewidth=1)
    ax1.legend(handles=bar_handles, loc="upper right", frameon=False)
    _style_axis(ax1)

    ax2.set_ylabel("best APEX score", color=MUTED_TEXT)
    _style_axis(ax2)

    fig.suptitle(
        f"milestone={milestone}, {task_set}: feasible frequency vs. best score, BOLT vs. ORPT (overlay)",
        color="#0b0b0b",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--milestone", type=int, required=True)
    parser.add_argument("--task-set", default="heldout20", choices=("trainset", "heldout20"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    df = pd.read_csv(results_dir / "per_peptide_alignment.csv")
    plots_dir = results_dir / "plots"

    for arm in ("BOLT", "ORPT"):
        plot_single_arm(df, arm, args.milestone, args.task_set, plots_dir / f"fig_peptide_alignment_{arm}.png")
    plot_overlay(df, args.milestone, args.task_set, plots_dir / "fig_peptide_alignment_overlay.png")


if __name__ == "__main__":
    main()
