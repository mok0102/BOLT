"""Shared colors and plotting style for synthetic Branin figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

ARM_COLOR: dict[str, str] = {
    # Proposed — red/coral family
    "ORPT":      "#E64B35B2",
    "ORPT-H0":   "#F39B7FB2",
    "ORPT-H1":   "#DC0000B2",
    "ORPT-MI":   "#A83232B2",

    # Baselines — no red/orange family
    "BOLT":      "#3C5488B2",
    "STBO":      "#00A087B2",
    "MTBO":      "#4DBBD5B2",
    "POGPE":     "#8491B4B2",
    "SGPE":      "#91D1C2B2",
    "OptFormer": "#7E6148B2",
    "LLAMBO":    "#6F6F6FB2",
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


def place_legend(ax, handles=None, labels=None) -> None:
    """The one fixed legend placement every module must use: outside the
    axes, upper-left anchored just past the right edge, so it never
    overlaps data regardless of curve shape."""
    kwargs = dict(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0.0)
    if handles is not None:
        ax.legend(handles, labels, **kwargs)
    else:
        ax.legend(**kwargs)


def savefig(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval2] wrote {path}")
    return path
