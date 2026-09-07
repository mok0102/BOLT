"""Shared plotting utilities for experiments/eval/'s plot_*.py scripts --
palette, axis styling, and the multi-results-dir CSV loading pattern.

Independent copy (not imported) of the equivalent module in
experiments/constraint_violation/ -- same conventions, kept as its own
module here so experiments/eval/ has no dependency on that directory.

Color convention: BOLT/ORPT keep their established blue/red; any other arm
name (e.g. ORPT-FA, ORPT-LEX) gets the next unused slot from a fixed
categorical palette, assigned deterministically by first appearance so
re-running with the same arm set always reproduces the same colors.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D

ARM_COLOR = {"BOLT": "#2a78d6", "ORPT": "#e34948"}
FALLBACK_COLORS = ["#1baf7a", "#eda100", "#4a3aa7", "#008300", "#eb6834", "#e87ba4", "#8c564b"]
GRID_COLOR = "#e1e0d9"
MUTED_TEXT = "#898781"


def arm_colors(arms: list[str]) -> dict[str, str]:
    """ARM_COLOR for known arms (BOLT/ORPT); any other arm name gets the next
    unused FALLBACK_COLORS slot, assigned in the order arms is given (caller
    should pass a stable, sorted arm list so this is reproducible)."""
    colors = dict(ARM_COLOR)
    next_fallback = 0
    for arm in arms:
        if arm not in colors:
            colors[arm] = FALLBACK_COLORS[next_fallback % len(FALLBACK_COLORS)]
            next_fallback += 1
    return colors


def sorted_arms_from_names(names: list[str]) -> list[str]:
    """BOLT, ORPT first (established convention), then any other arm names
    (e.g. ORPT-H0, ORPT-H1, ORPT-MI) alphabetically."""
    present = set(names)
    known = [a for a in ("BOLT", "ORPT") if a in present]
    other = sorted(present - set(known))
    return known + other


def sorted_arms(df: pd.DataFrame) -> list[str]:
    return sorted_arms_from_names(list(df["arm"].unique()))


def style_axis(ax) -> None:
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED_TEXT)


def filter_arms(df: pd.DataFrame, arms: list[str] | None) -> pd.DataFrame:
    """Restrict df to the given arm names (case-insensitive match against the
    "arm" column); returns df unchanged if arms is None/empty."""
    if not arms:
        return df
    wanted = {a.strip().upper() for a in arms}
    return df[df["arm"].str.upper().isin(wanted)]


def load_concat_csv(results_dirs: list[Path], filename: str) -> pd.DataFrame:
    """Reads `filename` from each results dir and concatenates them -- lets
    a comparison spanning several results dirs (e.g. re-run subsets) appear
    together in one chart."""
    frames = [pd.read_csv(d / filename) for d in results_dirs]
    return pd.concat(frames, ignore_index=True)


def plot_arm_line(
    ax, milestones: list[int], means: pd.Series, color: str,
    stds: pd.Series | None = None, label: str | None = None, marker_size: int = 8,
) -> None:
    """Plots one arm's per-milestone values. An arm with data at only a
    single milestone (e.g. LLAMBO, which is never fine-tuned -- its milestone
    value is an arbitrary placeholder, not a real training-progress
    checkpoint, see peptide_experiment/llambo_optimization.py) is drawn as a
    horizontal dashed reference line spanning the full milestone range
    instead of one point stranded at whichever placeholder milestone it was
    given: the value doesn't depend on milestone, so a line communicates that
    honestly instead of looking like missing data everywhere else.
    """
    present = means.dropna()
    if len(present) == 1:
        y = float(present.iloc[0])
        ax.plot([min(milestones), max(milestones)], [y, y], color=color, linewidth=2, linestyle="--", label=label)
        if stds is not None:
            std = stds.reindex(present.index).iloc[0]
            if pd.notna(std) and std > 0:
                ax.axhspan(y - std, y + std, color=color, alpha=0.15, linewidth=0)
        return
    ax.plot(milestones, means, color=color, linewidth=2, marker="o", markersize=marker_size, label=label)
    if stds is not None:
        ax.fill_between(
            milestones, means - stds.fillna(0), means + stds.fillna(0), color=color, alpha=0.15, linewidth=0,
        )


def is_milestone_independent(arm_sub: pd.DataFrame) -> bool:
    """True if this arm's rows only ever have a single milestone value (see
    plot_arm_line's docstring)."""
    return arm_sub["milestone"].nunique() == 1


def arm_legend_handles(arms: list[str], colors: dict[str, str], flat_arms: set[str]) -> list[Line2D]:
    """Proxy legend handles for figures that plot per-arm lines inside
    per-panel loops (so no single ax.plot call carries a label) -- matches
    plot_arm_line's solid-vs-dashed convention per arm."""
    return [
        Line2D([0], [0], color=colors[arm], linewidth=2.5, linestyle="--" if arm in flat_arms else "-")
        for arm in arms
    ]


def resolve_out_dir(results_dirs: list[Path], out_dir_arg: str | None) -> Path:
    """Default out-dir: the single --results-dir given, or results/comparison/
    next to this script if multiple were given."""
    if out_dir_arg:
        return Path(out_dir_arg)
    if len(results_dirs) == 1:
        return results_dirs[0]
    return Path(__file__).resolve().parent / "results" / "comparison"
