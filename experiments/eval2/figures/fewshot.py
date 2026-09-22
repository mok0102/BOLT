"""fig:fewshot -- best-of-k proposal quality vs. k
(number of valid unique proposal samples), one line per arm, log-x.

Reads summary_incumbent_vs_pool_size.csv only -- no config/manifest needed,
this experiment doesn't depend on oracle_budget at all.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from ..core.results_io import load_summary_incumbent
from ..core.spec import DOMAIN_SLUG
from . import style
from .labels import paper_arm


def generate(
    milestone: int,
    results_dirs: str | Path | list[str | Path],
    task_set: str,
    out_dir: Path,
) -> Path | None:
    df = load_summary_incumbent(results_dirs)
    if df.empty:
        print("[eval2.fewshot] no summary_incumbent_vs_pool_size.csv found, skipping")
        return None

    sub = df[(df["milestone"] == milestone) & (df["task_set"] == task_set)]
    if sub.empty:
        print(f"[eval2.fewshot] milestone={milestone} task_set={task_set}: no rows, skipping")
        return None

    style.apply_rcparams()
    fig, ax = plt.subplots(figsize=style.FIGSIZE)

    plotted_any = False
    for arm, arm_rows in sub.groupby("arm"):
        arm_rows = arm_rows.sort_values("n_proposals")
        label = paper_arm(arm)
        ax.plot(
            arm_rows["n_proposals"],
            arm_rows["mean_incumbent_mic"],
            color=style.arm_color(label),
            linewidth=style.LINEWIDTH,
            marker="o",
            markersize=style.MARKERSIZE,
            label=label,
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return None

    ax.set_xscale("log")
    style.style_axis(ax)
    ax.set_xlabel("Number of proposal samples (k)")
    ax.set_ylabel("Best-of-k objective")
    style.place_legend(ax)
    return style.savefig(fig, out_dir, f"fewshot_{DOMAIN_SLUG}")
