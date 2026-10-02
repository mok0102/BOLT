"""Shared visual style for this package's figure generators.

Ported from experiments/eval2/figures/style.py as it exists on
main-orpt-mol/main-orpt-peptide (via `git show`; main-orpt-branin carries no
experiments/eval2 source at all -- confirmed empty via `git ls-tree`), per
the user's explicit instruction to follow that package's figure format. Kept
byte-identical except for the one deliberate addition noted below
(motivational_exp.txt requires vector PDF output; the source `savefig()`
only produced PNG -- confirmed true of every figure script in this
repository, not just this one, so this is new work, not a copy).

Color convention: every arm this package ever plots gets ONE color, fixed
here, not assigned dynamically by first-appearance order. That's the whole
point -- BOLT/MTBO/etc. must look identical in every panel of Figures A/B/C.
All 8 arm names synthetic_experiment itself uses (BOLT, ORPT, STBO, MTBO,
POGPE, SGPE, OptFormer, LLAMBO) are already in ARM_COLOR below -- no
additions were needed.

CAUTION (carried over from the source file): these colors have not been
re-run through the dataviz skill's validate_palette.js in this environment.
ORPT's red is reserved -- no baseline arm may use a red/pink/red-orange hue.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ARM_COLOR: dict[str, str] = {
    # Proposed — red/coral family
    "ORPT":      "#E64B35B2",
    "ORPT-H0":   "#F39B7FB2",
    "ORPT-H1":   "#DC0000B2",
    "ORPT-H3":   "#A83232B2",
    "ORPT-MI":   "#A83232B2",

    # Baselines — no red/orange family
    "BOLT":      "#3C5488B2",
    "STBO":      "#00A087B2",
    "MTBO":      "#4DBBD5B2",
    "POGPE":     "#B09C85B2",
    "SGPE":      "#91D1C2B2",
    "OptFormer": "#7E6148B2",
    "LLAMBO":    "#6F6F6FB2",

    # Experiment C's simple/non-learned initializer baselines (core/
    # initializer_baselines.py) -- new to this package, not part of the
    # source file's roster. These three only ever co-occur with each other
    # plus BOLT/ORPT-H0/H1/H3 (figure_c.py's MAIN_METHODS), never with
    # STBO/MTBO/POGPE/SGPE/OptFormer/LLAMBO, so they only need to stay clear
    # of blue (BOLT) and the red family (ORPT) -- not the full roster above.
    # FIRST VERSION (now replaced): three tints/shades of one ColorBrewer
    # "Purples" sequential ramp (CBC9E2/9E9AC8/6A51A3). That palette is
    # built for gradients, where adjacent steps are SUPPOSED to look
    # similar -- exactly wrong for three arms meant to be told apart at a
    # glance, and it showed worst exactly where it mattered most: Random and
    # ContextRegression trace nearly the same curve (both decay from the
    # same initial regret), so color was the only remaining cue separating
    # them. Replaced with three genuinely different hues (gold / sea-green /
    # purple, roughly 90-145 degrees apart on the wheel) instead of one hue
    # at three lightnesses; ContextRegression keeps its original purple
    # since that one was never the problem.
    "Random":            "#C9A227B2",
    "PriorBestReuse":    "#2E8B57B2",
    "ContextRegression": "#6A51A3B2",
}
# Distinct linestyle per arm within the ORPT-H0/H1/H3 horizon-ablation group,
# since they share the "ORPT" red family and only co-occur in a filtered
# horizon-comparison view (Experiment A's main 6-method panel uses plain
# "ORPT" via labels.py::paper_arm, same convention as the source file's
# ORPT-H0/H1 ablation pair).
ABLATION_LINESTYLE: dict[str, str] = {
    "BOLT": "-",
    "ORPT-H0": ":",
    "ORPT-H1": "-",
    "ORPT-H3": "--",
}
# Same ablation group, for scatter-based figures (e.g. figure_c.py::
# generate_init_pools) where there is no line to carry a linestyle -- marker
# shape is the analogous secondary cue when points of different H land close
# enough together that color (same red family) isn't enough on its own.
ABLATION_MARKER: dict[str, str] = {
    "ORPT-H0": "^",
    "ORPT-H1": "o",
    "ORPT-H3": "s",
}

GRID_COLOR = "#e1e0d9"
MUTED_TEXT = "#898781"

# Wide enough for a readable x-axis when composed at roughly full text-width
# in the paper (not squeezed into a half-column slot), at print DPI.
FIGSIZE = (4.4, 2.75)
DPI = 300
LINEWIDTH = 1.8
MARKERSIZE = 4.5


def arm_color(arm: str) -> str:
    """Every arm this package plots must be in ARM_COLOR -- fail loudly on
    an unknown arm rather than silently falling back to a first-appearance
    slot (that's exactly the inconsistency this module exists to prevent)."""
    try:
        return ARM_COLOR[arm]
    except KeyError as e:
        raise KeyError(
            f"arm {arm!r} has no fixed color in style.ARM_COLOR -- add one "
            "explicitly, don't let it fall back to a dynamic slot"
        ) from e


def apply_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED_TEXT,
            "axes.labelcolor": "#1a1a1a",
            "text.color": "#1a1a1a",
            "xtick.color": MUTED_TEXT,
            "ytick.color": MUTED_TEXT,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.6,
            "axes.grid": True,
            "axes.axisbelow": True,
            "figure.figsize": FIGSIZE,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "legend.frameon": False,
        }
    )


def style_axis(ax) -> None:
    ax.grid(True, linewidth=0.6, color=GRID_COLOR)
    ax.set_axisbelow(True)


def place_legend(ax, handles=None, labels=None, fontsize=None) -> None:
    """The one fixed legend placement every module must use: outside the
    axes, upper-left anchored just past the right edge, so it never
    overlaps data regardless of curve shape. fontsize=None keeps
    rcParams["legend.fontsize"] (the shared default); pass an explicit size
    only for a figure with unusually many entries."""
    kwargs = dict(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0.0)
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    if handles is not None:
        ax.legend(handles, labels, **kwargs)
    else:
        ax.legend(**kwargs)


def savefig(fig, out_dir: Path, name: str) -> tuple[Path, Path]:
    """Writes BOTH a vector PDF and a high-resolution PNG (returns
    (pdf_path, png_path)) -- the one deliberate change from the source
    eval2/figures/style.py, whose own savefig() produced PNG only. Every
    figure script in this repository was checked (synthetic_experiment/
    eval2/style.py, experiments/eval2/figures/style.py on main-orpt-mol) and
    none currently emit PDF; motivational_exp.txt requires it ("vector PDF
    output, also export high-resolution PNG for inspection"), so this is new
    work, not something to have copied from an existing convention."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval2_branin_motivation] wrote {pdf_path}")
    print(f"[eval2_branin_motivation] wrote {png_path}")
    return pdf_path, png_path


def save_csv(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    """Write the exact rows a figure was plotted from, alongside its PDF/PNG
    (same stem, .csv extension) -- so a paper table can cite precise numbers
    without re-deriving them from the summary CSVs by hand. Byte-identical
    to the source file's save_csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"[eval2_branin_motivation] wrote {path}")
    return path
