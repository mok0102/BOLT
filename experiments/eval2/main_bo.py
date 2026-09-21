"""fig:main-bo -- one PNG per domain: best-objective-so-far vs. oracle calls
after initialization, one line per arm, at a single (usually final) milestone.

Scoped exception to this package's "fully independent" design (see the plan
file): correctly computing the running-best curve requires knowing which
trajectory rows are actually feasible (peptide: similarity to the task's
reference sequence >= cfg.similarity_threshold), which is real domain
science already implemented and verified in
experiments/eval/domains.py::Domain.running_best_series /
_running_best_series_from_mask. Reimplementing that independently here would
risk silently diverging from it (e.g. the first version of this file did:
it took a plain cummin over every row, including infeasible ones, which is
wrong). So main_bo.py imports domains.DOMAINS for that one piece only --
every other module in this package (fewshot/scaling/ablation) reads
already-computed summary CSVs and needs no domain science at all.

Matches experiments/eval/fig_main_bo.py's own convention: the plotted
y-values are raw train_y (domain.report_value's sign flip is NOT applied
here, same as the existing script -- report_value is only used by
incumbent_vs_pool_size.py for fig:fewshot's data).

Usage (see cli.py for the actual command-line entry point):
    generate("peptide", milestone=600, config_path=Path("peptide_experiment/configs/peptide_main_v2_eval_gpu3_lowbudget.yaml"),
             run_dirs={"BOLT": Path("runs/peptide_main_bolt_v2_lowbudget"),
                       "ORPT-H1": Path("runs/peptide_main_orpt_h1_v2_lowbudget"), ...},
             task_set="heldout100", out_dir=Path("experiments/eval2/out/peptide"))
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import style
from data import raw_trajectory_dir
from domain_specs import spec_for
from labels import paper_arm

_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))


def _arm_curve(domain_obj, cfg, run_dir: Path, task_set: str, arm: str, milestone: int, pool_size: int, oracle_budget: int) -> pd.Series | None:
    work_dir = raw_trajectory_dir(run_dir, task_set, arm, milestone, pool_size)
    task_ids = domain_obj.task_indices(cfg, task_set)
    series_list = []
    for task_id in task_ids:
        csv_path = work_dir / domain_obj.trajectory_csv_name(task_id)
        if not csv_path.exists():
            continue
        series = domain_obj.running_best_series(cfg, task_id, csv_path, pool_size)
        series_list.append(series.loc[:oracle_budget])
    if not series_list:
        print(f"[eval2.main_bo] {arm}-{milestone}: no usable task trajectories under {work_dir}, skipping")
        return None
    print(f"[eval2.main_bo] {arm}-{milestone}: {len(series_list)} tasks")
    return pd.concat(series_list, axis=1).mean(axis=1, skipna=True)


def generate(
    domain: str,
    milestone: int,
    config_path: Path,
    run_dirs: dict[str, Path],
    task_set: str,
    out_dir: Path,
) -> Path | None:
    from domains import DOMAINS  # local import: only resolvable once _EVAL_DIR is on sys.path

    spec = spec_for(domain)
    domain_obj = DOMAINS[domain]
    cfg = domain_obj.load_config(str(config_path))

    style.apply_rcparams()
    fig, ax = plt.subplots(figsize=style.FIGSIZE)

    plotted_any = False
    for arm, run_dir in run_dirs.items():
        curve = _arm_curve(domain_obj, cfg, run_dir, task_set, arm, milestone, spec.target_pool_size, spec.oracle_budget)
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
        print(f"[eval2.main_bo] domain={domain} milestone={milestone}: nothing to plot, skipping save")
        plt.close(fig)
        return None

    style.style_axis(ax)
    ax.set_xlabel("Oracle calls after initialization")
    ax.set_ylabel("Best objective found")
    style.place_legend(ax)
    return style.savefig(fig, out_dir, f"main_bo_{domain}")
