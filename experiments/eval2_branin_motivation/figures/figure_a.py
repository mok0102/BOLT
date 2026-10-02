"""Figure A: task scaling -- "where should cross-task knowledge enter BO?"

Reads the regret CSVs from core/regret.py (already joined against the
VERIFIED f_t^* file, core/global_optimum.py) and produces:

  motivation_A_task_scaling.pdf/png   -- 3-panel: b in {0, 10, 50} vs.
                                          completed training tasks
  motivation_A_trajectory_T50.pdf/png -- b=0..50 regret trajectory at T=50

Exact terminology from impl_plan/motivational_exp.txt, reproduced verbatim
(the doc is explicit that these strings must not be paraphrased):
  x-axis:  "Number of completed training tasks"
  y-axis:  "Simple regret to global optimum ↓"
  panel titles: "Initialization (0 BO calls)" / "After 10 BO calls" /
                "After 50 BO calls"
  trajectory x-axis: "Oracle calls after initialization"
"""

from __future__ import annotations

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
from core.regret import full_trajectory_table, regret_table  # noqa: E402
from synthetic_experiment.config import load_config  # noqa: E402

# The six methods Figure A contrasts as transfer interfaces (doc: "Do not
# frame the entire experiment as BOLT vs ORPT. ... BOLT/Top-K imitation is
# one relevant learned-initializer baseline."). STBO has no milestone axis
# (plotted as a flat reference). OptFormer/LLAMBO are supplementary, not
# plotted in the main 6-method panel (doc: "may be evaluated and retained in
# supplementary output").
MAIN_METHODS = ("STBO", "MTBO", "POGPE", "SGPE", "BOLT", "ORPT")
PANEL_B = (0, 10, 50)
PANEL_TITLE = {0: "Initialization (0 BO calls)", 10: "After 10 BO calls", 50: "After 50 BO calls"}


def _collect_scaling(
    baselines_cfg, orpt_cfg, milestones: list[int], heldout_tasks,
    init_size: int, oracle_budget: int,
) -> pd.DataFrame:
    """One row per (method, milestone, b, task_index). STBO is evaluated
    once (milestone=None) and broadcast across every milestone x-position
    for the scaling plot, since it has no training-task dependence -- its
    own regret numbers are never recomputed per milestone, only replotted at
    each x position, so it cannot look like it improved with more tasks."""
    rows = []
    for method in MAIN_METHODS:
        if method == "STBO":
            t = regret_table(baselines_cfg.run_dir, "STBO", None, heldout_tasks,
                             init_size, oracle_budget, list(PANEL_B))
            for milestone in milestones:
                broadcast = t.copy()
                broadcast["milestone"] = milestone
                rows.append(broadcast)
            continue
        cfg, run_dir = (orpt_cfg, orpt_cfg.run_dir) if method == "ORPT" else (baselines_cfg, baselines_cfg.run_dir)
        for milestone in milestones:
            t = regret_table(run_dir, method, milestone, heldout_tasks,
                             init_size, oracle_budget, list(PANEL_B))
            rows.append(t)
    data = pd.concat(rows, ignore_index=True)
    data.insert(0, "method", data["arm"].where(data["arm"] != "STBO", "STBO"))
    return data


def generate_task_scaling(
    baselines_config: str, orpt_config: str, out_dir: Path,
) -> tuple[Path, Path]:
    baselines_cfg = load_config(baselines_config)
    orpt_cfg = load_config(orpt_config)
    if baselines_cfg.manifest.token != orpt_cfg.manifest.token:
        raise ValueError("baselines and ORPT configs must share the same task manifest")
    milestones = baselines_cfg.milestones
    data = _collect_scaling(
        baselines_cfg, orpt_cfg, milestones, baselines_cfg.heldout_tasks,
        baselines_cfg.init_size, baselines_cfg.oracle_budget,
    )

    style.apply_rcparams()
    fig, axes = plt.subplots(1, 3, figsize=(style.FIGSIZE[0] * 2.6, style.FIGSIZE[1]), sharey=False)
    for ax, b in zip(axes, PANEL_B):
        panel = data[data.b == b]
        for method in MAIN_METHODS:
            series = panel[panel.method == method].groupby("milestone").simple_regret.agg(["mean", "sem"])
            series = series.reindex(milestones)
            # Log y-axis (see below): a lower error-bar limb that reaches <= 0
            # is undefined on a log scale, so it is clipped to a small
            # positive floor -- cosmetic only, the plotted MEAN is never
            # altered, only how far the lower whisker is drawn.
            lower_err = (series["mean"] - series["sem"]).clip(lower=1e-4)
            yerr = [series["mean"] - lower_err, series["sem"]]
            ax.errorbar(
                series.index, series["mean"], yerr=yerr,
                color=style.arm_color(method), marker="o" if method != "STBO" else None,
                linestyle="--" if method == "STBO" else "-",
                linewidth=style.LINEWIDTH, markersize=style.MARKERSIZE,
                capsize=2.0, elinewidth=0.8, label=method,
            )
        style.style_axis(ax)
        ax.set_title(PANEL_TITLE[b])
        ax.set_xlabel("Number of completed training tasks")
        ax.set_xticks(milestones)
        # Log scale: BOLT/ORPT/MTBO/STBO converge to within 2-3 orders of
        # magnitude of each other while POGPE/SGPE stay near their starting
        # regret -- on a linear axis the former cluster is indistinguishable
        # at this figure's size. Matches this repository's own precedent for
        # the identical problem (synthetic_experiment/eval2/plot_baselines.py
        # uses the same log-scale fix for STBO/BOLT/ORPT regret).
        ax.set_yscale("log")
    axes[0].set_ylabel("Simple regret to global optimum ↓")
    style.place_legend(axes[-1])
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    style.save_csv(data, out_dir, "motivation_A_task_scaling")
    return style.savefig(fig, out_dir, "motivation_A_task_scaling")


def generate_trajectory_t50(
    baselines_config: str, orpt_config: str, out_dir: Path, milestone: int = 50,
) -> tuple[Path, Path]:
    baselines_cfg = load_config(baselines_config)
    orpt_cfg = load_config(orpt_config)
    if milestone not in baselines_cfg.milestones:
        raise ValueError(f"milestone {milestone} not in {baselines_cfg.milestones}")

    rows = []
    for method in MAIN_METHODS:
        if method == "STBO":
            t = full_trajectory_table(baselines_cfg.run_dir, "STBO", None, baselines_cfg.heldout_tasks,
                                      baselines_cfg.init_size, baselines_cfg.oracle_budget)
        elif method == "ORPT":
            t = full_trajectory_table(orpt_cfg.run_dir, "ORPT", milestone, orpt_cfg.heldout_tasks,
                                      orpt_cfg.init_size, orpt_cfg.oracle_budget)
        else:
            t = full_trajectory_table(baselines_cfg.run_dir, method, milestone, baselines_cfg.heldout_tasks,
                                      baselines_cfg.init_size, baselines_cfg.oracle_budget)
        t.insert(0, "method", method)
        rows.append(t)
    data = pd.concat(rows, ignore_index=True)

    style.apply_rcparams()
    fig, ax = plt.subplots(figsize=style.FIGSIZE)
    for method in MAIN_METHODS:
        series = data[data.method == method].groupby("b").simple_regret.mean()
        ax.plot(series.index, series.values, color=style.arm_color(method),
                linestyle="--" if method == "STBO" else "-", linewidth=style.LINEWIDTH, label=method)
    style.style_axis(ax)
    ax.set_xlabel("Oracle calls after initialization")
    ax.set_ylabel("Simple regret to global optimum ↓")
    ax.set_xlim(0, baselines_cfg.oracle_budget)
    style.place_legend(ax)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    style.save_csv(data, out_dir, "motivation_A_trajectory_T50")
    return style.savefig(fig, out_dir, "motivation_A_trajectory_T50")
