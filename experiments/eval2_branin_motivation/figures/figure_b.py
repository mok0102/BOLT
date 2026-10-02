"""Figure B1/B2: standalone vs. rollout ranking (Experiment B).

Reads core/ranking_reversal.py's saved stats_{filtered,unfiltered}.json and
produces, per impl_plan/motivational_exp.txt's exact spec (strings
reproduced verbatim, not paraphrased):

  motivation_B_reversal.pdf/png
    x-axis: "Rollout horizon h"
    y-axis: "Ranking reversal rate (%)"
    title:  "Standalone vs. rollout ranking"
    annotation at h=TRAIN_HORIZON: "Training horizon H=1"

  motivation_B_horizon_agreement.pdf/png
    x-axis: "Evaluation horizon h"
    y-axis: "Ranking agreement with H=1 (%)"
    title:  "Short-horizon preference consistency"

Both panels plot WITH and WITHOUT the z-score reliability filter (doc:
"additionally report how results change with and without reliability
filtering") as two series, not just one -- the filtered series is the
one methodologically comparable to real ORPT pair construction, so it is
drawn solid/primary; unfiltered is the dashed/secondary reference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[1]
BOLT_ROOT = PKG_ROOT.parents[1]
for _p in (BOLT_ROOT, PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import style  # noqa: E402
from core.ranking_reversal import TRAIN_HORIZON  # noqa: E402

FILTER_STYLE = {
    True: {"linestyle": "-", "label_suffix": " (reliability-filtered)", "color": "#DC0000B2"},
    False: {"linestyle": "--", "label_suffix": " (unfiltered)", "color": "#898781B2"},
}


def _load(results_dir: Path) -> dict[bool, dict]:
    return {
        filt: json.loads((results_dir / f"stats_{'filtered' if filt else 'unfiltered'}.json").read_text())
        for filt in (True, False)
    }


def _as_frame(stats: dict, value_key: str, nested_key: str | None = None) -> pd.DataFrame:
    rows = []
    for h_str, row in stats["by_h"].items():
        source = row[nested_key] if nested_key else row
        if source is None:
            continue
        rows.append({"h": int(h_str), "value": source[value_key] * 100.0,
                     "ci95_low": source["ci95_low"] * 100.0, "ci95_high": source["ci95_high"] * 100.0})
    return pd.DataFrame(rows).sort_values("h")


def generate_reversal_figure(results_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    both = _load(results_dir)
    style.apply_rcparams()
    fig, ax = plt.subplots(figsize=style.FIGSIZE)
    all_rows = []
    for filt in (True, False):
        frame = _as_frame(both[filt], "reversal_rate")
        frame["reliability_filter"] = filt
        all_rows.append(frame)
        fstyle = FILTER_STYLE[filt]
        ax.errorbar(
            frame.h, frame.value, yerr=[frame.value - frame.ci95_low, frame.ci95_high - frame.value],
            color=fstyle["color"], linestyle=fstyle["linestyle"], linewidth=style.LINEWIDTH,
            marker="o", markersize=style.MARKERSIZE, capsize=2.0, elinewidth=0.8,
            label=f"Reversal rate{fstyle['label_suffix']}",
        )
    style.style_axis(ax)
    ax.set_xlabel("Rollout horizon h")
    ax.set_ylabel("Ranking reversal rate (%)")
    # pad: the required y-label text is long enough that its rotated top end
    # reaches the axes' top-left corner, where a title at the default pad
    # collides with it -- extra clearance, not a text change (both strings
    # are the doc's required verbatim text).
    ax.set_title("Standalone vs. rollout ranking", pad=14)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 50])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    # Log scale draws its own decade minor ticks (1, 10, 100, ...) by default,
    # which collide with and duplicate the explicit major ticks above once the
    # tick set doesn't itself span whole decades evenly -- turn them off
    # rather than let the two tick layers overlap.
    ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.minorticks_off()
    ymin, ymax = ax.get_ylim()
    ax.axvline(TRAIN_HORIZON, color=style.MUTED_TEXT, linewidth=1.0, linestyle=":")
    ax.annotate(
        "Training horizon H=1", xy=(TRAIN_HORIZON, ymax), xytext=(TRAIN_HORIZON * 1.15, ymax * 0.92),
        fontsize=7, color=style.MUTED_TEXT,
    )
    style.place_legend(ax)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    style.save_csv(pd.concat(all_rows, ignore_index=True), out_dir, "motivation_B_reversal")
    return style.savefig(fig, out_dir, "motivation_B_reversal")


def generate_agreement_figure(results_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    both = _load(results_dir)
    style.apply_rcparams()
    fig, ax = plt.subplots(figsize=style.FIGSIZE)
    all_rows = []
    for filt in (True, False):
        frame = _as_frame(both[filt], "agreement_rate", nested_key="agreement_with_Htrain")
        frame["reliability_filter"] = filt
        all_rows.append(frame)
        fstyle = FILTER_STYLE[filt]
        ax.errorbar(
            frame.h, frame.value, yerr=[frame.value - frame.ci95_low, frame.ci95_high - frame.value],
            color=fstyle["color"], linestyle=fstyle["linestyle"], linewidth=style.LINEWIDTH,
            marker="o", markersize=style.MARKERSIZE, capsize=2.0, elinewidth=0.8,
            label=f"Agreement with H=1{fstyle['label_suffix']}",
        )
    style.style_axis(ax)
    ax.set_xlabel("Evaluation horizon h")
    ax.set_ylabel("Ranking agreement with H=1 (%)")
    # Same title/ylabel clearance fix as generate_reversal_figure.
    ax.set_title("Short-horizon preference consistency", pad=14)
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 10, 20, 50])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    # See generate_reversal_figure's identical fix: log scale's own decade
    # minor ticks (1, 10, 100) collide with and duplicate these explicit ones.
    ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.minorticks_off()
    style.place_legend(ax)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    style.save_csv(pd.concat(all_rows, ignore_index=True), out_dir, "motivation_B_horizon_agreement")
    return style.savefig(fig, out_dir, "motivation_B_horizon_agreement")
