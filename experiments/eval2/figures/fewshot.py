"""fig:fewshot -- best-of-k proposal quality vs. k
(number of valid unique proposal samples), one line per arm, log-x.

Reads summary_incumbent_vs_pool_size.csv only -- no config/manifest needed,
this experiment doesn't depend on oracle_budget at all. reference_lines is
the one exception: self-seeding arms (STBO/MTBO/POGPE/SGPE) have no "raw
proposal, best-of-k" curve of their own (they build a pool by mutation, not
LLM sampling), so instead of a curve they get a single horizontal dotted
line at their own final-BO outcome, computed by the caller (main_bo.py's
raw-trajectory helpers, same as main_results_table.py) and passed in
pre-computed -- this module still never touches oracle_budget/cfg itself.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ..core.results_io import load_summary_incumbent
from ..core.spec import DOMAIN_SLUG
from . import style
from .labels import paper_arm


def generate(
    milestone: int,
    results_dirs: str | Path | list[str | Path],
    task_set: str,
    out_dir: Path,
    reference_lines: list[tuple[str, str, float]] | None = None,  # (arm, label, final_bo_value)
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
        label = f"{paper_arm(arm)}-{milestone}"
        ax.plot(
            arm_rows["n_proposals"],
            arm_rows["mean_incumbent_mic"],
            color=style.arm_color(paper_arm(arm)),
            linewidth=style.LINEWIDTH,
            marker="o",
            markersize=style.MARKERSIZE,
            label=label,
        )
        plotted_any = True

    for arm, label, value in reference_lines or []:
        ax.axhline(value, color=style.arm_color(paper_arm(arm)), linestyle=":", linewidth=style.LINEWIDTH, label=label)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return None

    ax.set_yscale("log")
    style.style_axis(ax)
    ax.set_xlabel("Number of proposal samples (k)")
    ax.set_ylabel("Best-of-k MIC")
    style.place_legend(ax)

    name = f"fewshot_{DOMAIN_SLUG}"
    export = sub[["n_proposals", "arm", "mean_incumbent_mic"]].copy()
    export["arm"] = export["arm"].map(paper_arm)
    export = export.rename(columns={"n_proposals": "Number of proposal samples", "arm": "model", "mean_incumbent_mic": "y"})
    export["kind"] = "proposal_quality"
    ref_rows = pd.DataFrame([
        {"Number of proposal samples": None, "model": label, "y": value, "kind": "final_bo_reference"}
        for _arm, label, value in (reference_lines or [])
    ])
    export = pd.concat([export, ref_rows], ignore_index=True).sort_values(["kind", "model", "Number of proposal samples"]).reset_index(drop=True)
    style.save_csv(export, out_dir, name)
    return style.savefig(fig, out_dir, name)
