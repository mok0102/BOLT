"""fig:scaling -- two separate PNGs (fully atomized, no internal sub-panels):
"init" (best-of-k at a representative k vs. completed training tasks) and
"finalbo" (final BO objective at the full oracle budget vs. completed
training tasks).

target_pool_size / oracle_budget are not hardcoded here: they are read off
the summary CSV the run actually produced, so these figures can never
disagree with the experiment that generated them. Pass them explicitly to
pin a specific slice when a results dir holds more than one.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ..core.results_io import load_summary_fixed_target_bo, load_summary_incumbent
from ..core.spec import DOMAIN_SLUG, SCALING_K
from . import style
from .labels import paper_arm


def _line_plot(ax, df, x_col, y_col, label_col="arm"):
    plotted_any = False
    for arm, arm_rows in df.groupby(label_col):
        arm_rows = arm_rows.sort_values(x_col)
        label = paper_arm(arm)
        ax.plot(
            arm_rows[x_col],
            arm_rows[y_col],
            color=style.arm_color(label),
            linewidth=style.LINEWIDTH,
            marker="o",
            markersize=style.MARKERSIZE,
            label=label,
        )
        plotted_any = True
    return plotted_any


def _pick(df, column: str, requested: int | None, what: str) -> int | None:
    """Resolve a slice value: the caller's if given, else the data's own --
    a single unique value, or the max when the column is a budget axis."""
    if requested is not None:
        return requested
    values = sorted(df[column].dropna().unique())
    if not values:
        return None
    if len(values) > 1:
        print(f"[eval2.scaling] {what}: {column} has {values}, using {values[-1]} -- pass it explicitly to pin another")
    return int(values[-1])


def generate_init(
    results_dirs: str | Path | list[str | Path],
    task_set: str,
    out_dir: Path,
    scaling_k: int = SCALING_K,
) -> Path | None:
    df = load_summary_incumbent(results_dirs)
    if df.empty:
        print("[eval2.scaling] init: no summary_incumbent_vs_pool_size.csv, skipping")
        return None
    sub = df[(df["task_set"] == task_set) & (df["n_proposals"] == scaling_k)]
    if sub.empty:
        print(f"[eval2.scaling] init: no rows at k={scaling_k}, skipping")
        return None

    style.apply_rcparams()
    fig, ax = plt.subplots(figsize=style.FIGSIZE)
    if not _line_plot(ax, sub, "milestone", "mean_incumbent_mic"):
        plt.close(fig)
        return None
    style.style_axis(ax)
    ax.set_xlabel("Completed training tasks")
    ax.set_ylabel(f"Best-of-{scaling_k} objective")
    style.place_legend(ax)
    return style.savefig(fig, out_dir, f"scaling_{DOMAIN_SLUG}_init")


def generate_finalbo(
    results_dirs: str | Path | list[str | Path],
    task_set: str,
    out_dir: Path,
    target_pool_size: int | None = None,
    oracle_budget: int | None = None,
) -> Path | None:
    df = load_summary_fixed_target_bo(results_dirs)
    if df.empty:
        print("[eval2.scaling] finalbo: no summary_fixed_target_bo.csv, skipping")
        return None
    df = df[df["task_set"] == task_set]
    if df.empty:
        print(f"[eval2.scaling] finalbo: no rows for task_set={task_set}, skipping")
        return None

    target = _pick(df, "target_pool_size", target_pool_size, "finalbo")
    budget = _pick(df, "bo_calls", oracle_budget, "finalbo")
    sub = df[(df["target_pool_size"] == target) & (df["bo_calls"] == budget)]
    if sub.empty:
        print(f"[eval2.scaling] finalbo: no rows at target={target} bo_calls={budget}, skipping")
        return None

    style.apply_rcparams()
    fig, ax = plt.subplots(figsize=style.FIGSIZE)
    if not _line_plot(ax, sub, "milestone", "mean_best_mic"):
        plt.close(fig)
        return None
    style.style_axis(ax)
    ax.set_xlabel("Completed training tasks")
    ax.set_ylabel("Final BO objective")
    style.place_legend(ax)
    return style.savefig(fig, out_dir, f"scaling_{DOMAIN_SLUG}_finalbo")
