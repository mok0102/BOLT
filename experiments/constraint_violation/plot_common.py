"""Shared plotting utilities for this directory's comparison charts
(plot_violation_rate.py, plot_table11.py, plot_rejection_sampled_bo.py,
plot_incumbent_curve.py) -- palette, axis styling, and the multi-results-dir
CSV loading pattern, previously copy-pasted identically across all of them.

Color convention: BOLT/ORPT keep their established blue/red; any other arm
name (e.g. ORPT-LEX, a future ORPT-soft-margin) gets the next unused slot
from the dataviz skill's fixed categorical order (aqua, yellow, violet,
green, orange, magenta), assigned deterministically by first appearance so
re-running with the same arm set always reproduces the same colors.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ARM_COLOR = {"BOLT": "#2a78d6", "ORPT": "#e34948"}
FALLBACK_COLORS = ["#1baf7a", "#eda100", "#4a3aa7", "#008300", "#eb6834", "#e87ba4"]
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


def sorted_arms(df: pd.DataFrame) -> list[str]:
    """BOLT, ORPT first (established convention), then any other arm names
    (e.g. ORPT-LEX) alphabetically."""
    present = set(df["arm"].unique())
    known = [a for a in ("BOLT", "ORPT") if a in present]
    other = sorted(present - set(known))
    return known + other


def style_axis(ax) -> None:
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED_TEXT)


def load_concat_csv(results_dirs: list[Path], filename: str) -> pd.DataFrame:
    """Reads `filename` from each results dir (one per experiment_id) and
    concatenates them -- lets arms from different experiments (e.g. BOLT/ORPT
    from one, ORPT-LEX from another) appear together in one comparison."""
    frames = [pd.read_csv(d / filename) for d in results_dirs]
    return pd.concat(frames, ignore_index=True)


def resolve_out_dir(results_dirs: list[Path], out_dir_arg: str | None) -> Path:
    """Default out-dir: the single --results-dir given, or results/comparison/
    next to this script if multiple were given (matches every plot_*.py's
    --out-dir convention)."""
    if out_dir_arg:
        return Path(out_dir_arg)
    if len(results_dirs) == 1:
        return results_dirs[0]
    return Path(__file__).resolve().parent / "results" / "comparison"
