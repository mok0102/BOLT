"""fig:main-bo -- best-objective-so-far vs. oracle calls after initialization,
one line per arm, at a single (usually final) milestone.

Unlike the other figures here, this one does not read a summary CSV: it
replays the dense per-task BO trajectories compute/fixed_target_bo.py wrote
and averages them. Deciding which trajectory rows even count requires the
feasibility check that only the domain knows, so this module calls
domains.peptide.running_best_series rather than reducing the CSV itself --
an earlier version took a plain cummin over every row, including infeasible
ones, and was wrong.

Plotted y-values are the domain's own reported objective (MIC for peptide),
matching what compute/incumbent.py records for fig:fewshot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..core.arms import ArmSpec, bo_work_dir_for
from ..core.spec import DOMAIN_SLUG
from ..domains import peptide
from . import style
from .labels import paper_arm


def _arm_curve(cfg, run_dir: Path, task_set: str, arm: str, milestone: int, pool_size: int, oracle_budget: int) -> pd.Series | None:
    work_dir = bo_work_dir_for(ArmSpec(arm=arm, milestone=milestone, run_dir=run_dir), task_set, pool_size)
    series_list = []
    for task_id in peptide.task_indices(cfg, task_set):
        csv_path = work_dir / peptide.trajectory_csv_name(task_id)
        if not csv_path.exists():
            continue
        series = peptide.running_best_series(cfg, task_id, csv_path, pool_size)
        series_list.append(series.loc[:oracle_budget])
    if not series_list:
        print(f"[eval2.main_bo] {arm}-{milestone}: no usable task trajectories under {work_dir}, skipping")
        return None
    print(f"[eval2.main_bo] {arm}-{milestone}: {len(series_list)} tasks")
    return pd.concat(series_list, axis=1).mean(axis=1, skipna=True)


def generate(
    milestone: int,
    config_path: Path,
    run_dirs: dict[str, Path],
    task_set: str,
    out_dir: Path,
    target_pool_size: int | None = None,
) -> Path | None:
    cfg = peptide.load_config(str(config_path))
    pool_size = target_pool_size if target_pool_size is not None else cfg.init_size

    style.apply_rcparams()
    fig, ax = plt.subplots(figsize=style.FIGSIZE)

    plotted_any = False
    for arm, run_dir in run_dirs.items():
        curve = _arm_curve(cfg, run_dir, task_set, arm, milestone, pool_size, cfg.oracle_budget)
        if curve is None:
            continue
        ax.plot(
            curve.index,
            curve.to_numpy(),
            color=style.arm_color(paper_arm(arm)),
            linewidth=style.LINEWIDTH,
            label=paper_arm(arm),
        )
        plotted_any = True

    if not plotted_any:
        print(f"[eval2.main_bo] milestone={milestone}: nothing to plot, skipping save")
        plt.close(fig)
        return None

    style.style_axis(ax)
    ax.set_xlabel("Oracle calls after initialization")
    ax.set_ylabel("Best objective found")
    style.place_legend(ax)
    return style.savefig(fig, out_dir, f"main_bo_{DOMAIN_SLUG}")
