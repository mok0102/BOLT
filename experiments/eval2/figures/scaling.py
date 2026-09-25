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
import pandas as pd

from ..core.results_io import load_per_task_fixed_target_bo, load_summary_incumbent
from ..core.spec import DOMAIN_SLUG, SCALING_K
from . import style
from .labels import paper_arm


def _line_plot(ax, df, x_col, y_col, label_col="arm"):
    plotted_any = False
    for arm, arm_rows in df.groupby(label_col):
        arm_rows = arm_rows.sort_values(x_col)
        label = paper_arm(arm)
        color = style.arm_color(label)
        if len(arm_rows) == 1:
            # A single row means this arm doesn't actually vary with x_col
            # (e.g. STBO/LLAMBO, which only ever run at the milestone-
            # independent placeholder x=0 -- see arm_specs/main.yaml's own
            # comment). One point would just look like an isolated dot far
            # from the other arms' range, so draw it as a constant reference
            # line spanning the axis instead, dashed to mark it as a
            # baseline rather than a real x_col trend.
            ax.axhline(
                arm_rows[y_col].iloc[0],
                color=color,
                linewidth=style.LINEWIDTH,
                linestyle="--",
                label=label,
            )
        else:
            ax.plot(
                arm_rows[x_col],
                arm_rows[y_col],
                color=color,
                linewidth=style.LINEWIDTH,
                marker="o",
                markersize=style.MARKERSIZE,
                label=label,
            )
        plotted_any = True
    return plotted_any


def _export_csv(df, x_col, y_col, x_label):
    """The exact rows behind a scaling figure, in the fixed 3-column shape
    (x_label, model, y) both scaling figures share -- so per-milestone
    numbers can be cited in a paper table without re-deriving them from the
    summary CSV by hand."""
    out = df[[x_col, "arm", y_col]].copy()
    out["arm"] = out["arm"].map(paper_arm)
    out = out.rename(columns={x_col: x_label, "arm": "model", y_col: "y"})
    return out.sort_values(["model", x_label]).reset_index(drop=True)


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
    name = f"scaling_{DOMAIN_SLUG}_init"
    style.save_csv(_export_csv(sub, "milestone", "mean_incumbent_mic", "completed training task"), out_dir, name)
    return style.savefig(fig, out_dir, name)


def _intersect_and_mean(df, y_col: str) -> pd.DataFrame:
    """Per milestone, restrict every arm's mean to the SAME task_idx subset
    (the intersection across arms present at that milestone) instead of each
    arm's own coverage.

    Coverage varies a lot by arm: self-seeding arms (MTBO/OptFormer/STBO/
    LLAMBO) always cover every task, but BOLT/ORPT-H1's real rejection-
    sampled pool means some tasks never reach target_pool_size feasible
    candidates -- worse at higher milestones (e.g. BOLT-600 covered only
    18/50 heldout50 tasks). Averaging each arm over its own coverage set
    compares different task subsets and can make an arm look worse for a
    reason that has nothing to do with its BO performance. STBO/LLAMBO's
    milestone=0 rows form their own group here (untouched by the real-
    milestone arms' intersection) since they're a single milestone-
    independent reference line, not a per-milestone series.
    """
    rows = []
    for milestone, group in df.groupby("milestone"):
        task_sets_by_arm = {arm: set(arm_rows["task_idx"]) for arm, arm_rows in group.groupby("arm")}
        shared = set.intersection(*task_sets_by_arm.values())
        if not shared:
            print(f"[eval2.scaling] finalbo milestone={milestone}: no tasks shared by every arm, skipping")
            continue
        sizes = ", ".join(f"{arm}={len(s)}" for arm, s in sorted(task_sets_by_arm.items()))
        print(f"[eval2.scaling] finalbo milestone={milestone}: intersection={len(shared)} tasks (own coverage: {sizes})")
        restricted = group[group["task_idx"].isin(shared)]
        for arm, arm_rows in restricted.groupby("arm"):
            rows.append({"milestone": milestone, "arm": arm, y_col: arm_rows[y_col].mean()})
    return pd.DataFrame(rows)


def generate_finalbo(
    results_dirs: str | Path | list[str | Path],
    task_set: str,
    out_dir: Path,
    target_pool_size: int | None = None,
    oracle_budget: int | None = None,
) -> Path | None:
    df = load_per_task_fixed_target_bo(results_dirs)
    if df.empty:
        print("[eval2.scaling] finalbo: no per_task_fixed_target_bo.csv, skipping")
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
    sub = _intersect_and_mean(sub, "best_mic")
    if sub.empty:
        print("[eval2.scaling] finalbo: no milestone had a shared task across all its arms, skipping")
        return None

    style.apply_rcparams()
    fig, ax = plt.subplots(figsize=style.FIGSIZE)
    if not _line_plot(ax, sub, "milestone", "best_mic"):
        plt.close(fig)
        return None
    style.style_axis(ax)
    ax.set_xlabel("Completed training tasks")
    ax.set_ylabel("Final BO objective")
    style.place_legend(ax)
    name = f"scaling_{DOMAIN_SLUG}_finalbo"
    style.save_csv(_export_csv(sub, "milestone", "best_mic", "completed training task"), out_dir, name)
    return style.savefig(fig, out_dir, name)
