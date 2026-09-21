"""fig:scaling -- two separate PNGs per domain (fully atomized, no internal
sub-panels): "init" (best-of-k at a representative k vs. completed training
tasks) and "finalbo" (final BO objective at the full oracle budget vs.
completed training tasks).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import style
from data import load_summary_fixed_target_bo, load_summary_incumbent
from domain_specs import spec_for
from labels import paper_arm


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


def generate_init(
    domain: str,
    results_dirs: str | Path | list[str | Path],
    task_set: str,
    out_dir: Path,
) -> Path | None:
    spec = spec_for(domain)
    df = load_summary_incumbent(results_dirs)
    if df.empty:
        print(f"[eval2.scaling] domain={domain} init: no summary_incumbent_vs_pool_size.csv, skipping")
        return None
    sub = df[(df["task_set"] == task_set) & (df["n_proposals"] == spec.scaling_k)]
    if sub.empty:
        print(f"[eval2.scaling] domain={domain} init: no rows at k={spec.scaling_k}, skipping")
        return None

    style.apply_rcparams()
    fig, ax = plt.subplots(figsize=style.FIGSIZE)
    if not _line_plot(ax, sub, "milestone", "mean_incumbent_mic"):
        plt.close(fig)
        return None
    style.style_axis(ax)
    ax.set_xlabel("Completed training tasks")
    ax.set_ylabel(f"Best-of-{spec.scaling_k} objective")
    style.place_legend(ax)
    return style.savefig(fig, out_dir, f"scaling_{domain}_init")


def generate_finalbo(
    domain: str,
    results_dirs: str | Path | list[str | Path],
    task_set: str,
    out_dir: Path,
) -> Path | None:
    spec = spec_for(domain)
    df = load_summary_fixed_target_bo(results_dirs)
    if df.empty:
        print(f"[eval2.scaling] domain={domain} finalbo: no summary_fixed_target_bo.csv, skipping")
        return None
    sub = df[
        (df["task_set"] == task_set)
        & (df["target_pool_size"] == spec.target_pool_size)
        & (df["bo_calls"] == spec.oracle_budget)
    ]
    if sub.empty:
        print(f"[eval2.scaling] domain={domain} finalbo: no rows at bo_calls={spec.oracle_budget}, skipping")
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
    return style.savefig(fig, out_dir, f"scaling_{domain}_finalbo")
