"""For one example task (one target peptide), plot the rejection-sampled
candidate distribution: x-axis = distinct surviving peptide (sorted by how
often it was resampled, descending), left y-axis = sampling frequency (bar),
right y-axis = APEX score (line). One panel per arm (BOLT-<m> | ORPT-<m>)
side by side so the same task's two candidate pools -- which are drawn
independently and don't share any x-axis identity -- can still be compared
by eye.

Genuine dual-axis chart by explicit request (frequency and unbounded score
sharing one panel) -- normally an anti-pattern per the dataviz skill, but
built as asked rather than substituted.

Produces one figure per requested task_idx from
peptide_frequency_alignment.py's per_sequence_frequency_alignment.csv /
per_task_frequency_alignment_summary.csv.

Usage (run from the BOLT repo root):
    python experiments/constraint_violation/plot_peptide_frequency_alignment.py \\
        --results-dir experiments/constraint_violation/results/peptide_100task_orpt_beta0.25 \\
        --milestone 100 --task-set heldout20 --task-idx 900
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_common import ARM_COLOR, GRID_COLOR, MUTED_TEXT

DEFAULT_TOP_N = 30


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.tick_params(colors=MUTED_TEXT)


def plot_task(
    seq_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    task_idx: int,
    milestone: int,
    task_set: str,
    top_n: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_bar, arm in zip(axes, ("BOLT", "ORPT")):
        sub = seq_df[
            (seq_df["arm"] == arm) & (seq_df["milestone"] == milestone)
            & (seq_df["task_set"] == task_set) & (seq_df["task_idx"] == task_idx)
        ].sort_values("freq_rank").head(top_n)

        summary = summary_df[
            (summary_df["arm"] == arm) & (summary_df["milestone"] == milestone)
            & (summary_df["task_set"] == task_set) & (summary_df["task_idx"] == task_idx)
        ]

        if sub.empty:
            ax_bar.set_title(f"{arm}-{milestone}: no data", color=MUTED_TEXT)
            ax_bar.axis("off")
            continue

        x = range(1, len(sub) + 1)
        ax_bar.bar(x, sub["frequency"], color=ARM_COLOR[arm], alpha=0.55)
        ax_bar.set_ylabel("sampling frequency (count)", color=MUTED_TEXT)
        ax_bar.set_xlabel("distinct peptide, sorted by sampling frequency", color=MUTED_TEXT)
        ax_bar.grid(True, axis="y", color=GRID_COLOR, linewidth=1)
        _style_axis(ax_bar)

        ax_line = ax_bar.twinx()
        ax_line.plot(x, sub["score"], color=ARM_COLOR[arm], linewidth=2, marker="o", markersize=4)
        ax_line.set_ylabel("APEX score", color=MUTED_TEXT)
        _style_axis(ax_line)

        title = f"{arm}-{milestone}"
        if not summary.empty:
            row = summary.iloc[0]
            title += (
                f"\n{int(row['n_accepted_draws'])}/{int(row['n_total_draws'])} draws accepted "
                f"({row['rejection_rate']:.0%} rejected), {int(row['n_unique_feasible'])} unique feasible"
            )
        ax_bar.set_title(title, color="#0b0b0b")

    fig.suptitle(
        f"Task {task_idx} (milestone={milestone}, {task_set}): rejection-sampled frequency vs. APEX score",
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
    parser.add_argument(
        "--task-idx",
        default=None,
        help="Comma-separated task indices to plot, one figure each (default: the first task "
        "found in per_sequence_frequency_alignment.csv for this milestone/task_set)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Only show the top N most-frequently-resampled peptides per arm (default {DEFAULT_TOP_N})",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    seq_df = pd.read_csv(results_dir / "per_sequence_frequency_alignment.csv")
    summary_df = pd.read_csv(results_dir / "per_task_frequency_alignment_summary.csv")
    plots_dir = results_dir / "plots"

    if args.task_idx is not None:
        task_indices = [int(t.strip()) for t in args.task_idx.split(",") if t.strip()]
    else:
        candidates = seq_df[
            (seq_df["milestone"] == args.milestone) & (seq_df["task_set"] == args.task_set)
        ]["task_idx"]
        if candidates.empty:
            raise ValueError(f"No rows for milestone={args.milestone}, task_set={args.task_set}")
        task_indices = [int(candidates.iloc[0])]

    for task_idx in task_indices:
        plot_task(
            seq_df,
            summary_df,
            task_idx,
            args.milestone,
            args.task_set,
            args.top_n,
            plots_dir / f"fig_frequency_alignment_task{task_idx}.png",
        )


if __name__ == "__main__":
    main()
