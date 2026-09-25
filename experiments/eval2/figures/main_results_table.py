"""tab:main-results (peptide half) -- one CSV + one bare tabular* snippet
mirroring ablation.py's csv+tex twin, but reading raw per-task BO
trajectories directly (peptide.running_best_series) rather than a summary
CSV: runs/lorarank_8's fixed_target_bo output was produced by a one-off
parallel sweep script that never wrote summary_fixed_target_bo.csv /
per_task_fixed_target_bo.csv, only the dense per-task task_XXXX.csv files
main_bo.py's fig1b already reads -- see main_bo._available_task_ids.

Each row picks whatever milestone/n_experts that arm's data actually has;
see ROW_SPECS below for which checkpoint each one is pinned to and why.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.spec import DOMAIN_SLUG
from ..domains import peptide
from .main_bo import _available_task_ids

# (arm, milestone, run_dir_key, row_label). run_dir_key selects bolt_run_dir
# vs orpt_run_dir (below); row_label is the paper's tab:main-results method
# name, suffixed with -<milestone> for BOLT/ORPT-MI since they get TWO rows
# each (630 and 900) instead of one. Both are reported deliberately: BOLT and
# ORPT-MI converge at milestone=900 (visible in the boltorpt figure's own
# 126-900 sweep), and showing only 630 (where the gap is largest) while
# MTBO/OptFormer are stuck at 900 (their only available checkpoint -- 378/630
# data does not exist for these two arms) would selectively hide that
# convergence. POGPE/SGPE use n_experts=20; STBO has no milestone axis at all.
ROW_SPECS: list[tuple[str, int, str, str]] = [
    ("BOLT", 630, "bolt", "BOLT-630"),
    ("BOLT", 900, "bolt", "BOLT-900"),
    ("ORPT-MI", 630, "orpt", "ORPT-630"),
    ("ORPT-MI", 900, "orpt", "ORPT-900"),
    ("STBO", 0, "bolt", "STBO"),
    ("MTBO", 900, "bolt", "MTBO"),
    ("POGPE", 20, "bolt", "POGPE"),
    ("SGPE", 20, "bolt", "SGPE"),
    ("OptFormer", 900, "bolt", "OptFormer"),
]
# LLAMBO has no raw/BO data anywhere under runs/lorarank_8 (never run in this
# sweep) -- included as an unfilled row so the table shape still matches
# tab:main-results, per that table's own "dashes are unfilled placeholders"
# convention.
NO_DATA_ROWS: list[str] = ["LLAMBO"]


def _per_task_endpoints(cfg, work_dir: Path, task_ids, oracle_budget: int) -> tuple[list[float], list[float]]:
    """Per task: the value at oracle_calls=0 (Initialization, the built init
    pool's own best) and at oracle_calls=oracle_budget (Final BO). A task
    whose trajectory ended early has no entry at oracle_budget (see
    running_best_series's own docstring) and is simply excluded from that
    column, not filled in with a stale earlier value."""
    inits: list[float] = []
    finals: list[float] = []
    for task_id in task_ids:
        csv_path = work_dir / peptide.trajectory_csv_name(task_id)
        # Per-task actual pool size (see main_bo.py's _arm_curve for why):
        # a "fixed budget" task's real init pool can be smaller than pool_size.
        real_pool_size = peptide.read_pool_size(work_dir, task_id)
        series = peptide.running_best_series(cfg, task_id, csv_path, real_pool_size)
        init_val = series.get(0)
        if init_val is not None and pd.notna(init_val):
            inits.append(float(init_val))
        final_val = series.get(oracle_budget)
        if final_val is not None and pd.notna(final_val):
            finals.append(float(final_val))
    return inits, finals


def build_rows(cfg, bolt_run_dir: Path, orpt_run_dir: Path, task_set: str, pool_size: int) -> list[dict]:
    run_dirs = {"bolt": bolt_run_dir, "orpt": orpt_run_dir}
    rows: list[dict] = []
    for arm, milestone, run_dir_key, label in ROW_SPECS:
        run_dir = run_dirs[run_dir_key]
        work_dir, task_ids = _available_task_ids(cfg, run_dir, task_set, arm, milestone, pool_size)
        if not task_ids:
            print(f"[eval2.main_results_table] {arm}-{milestone}: no usable trajectories, leaving row empty")
            rows.append({"method": label, "checkpoint_or_n": milestone,
                         "n_tasks_init": 0, "init_mean": None, "init_std": None,
                         "n_tasks_final": 0, "final_mean": None, "final_std": None})
            continue
        inits, finals = _per_task_endpoints(cfg, work_dir, sorted(task_ids), cfg.oracle_budget)
        init_s, final_s = pd.Series(inits, dtype=float), pd.Series(finals, dtype=float)
        rows.append({
            "method": label, "checkpoint_or_n": milestone,
            "n_tasks_init": len(init_s), "init_mean": init_s.mean() if len(init_s) else None,
            "init_std": init_s.std() if len(init_s) > 1 else None,
            "n_tasks_final": len(final_s), "final_mean": final_s.mean() if len(final_s) else None,
            "final_std": final_s.std() if len(final_s) > 1 else None,
        })
    for label in NO_DATA_ROWS:
        rows.append({"method": label, "checkpoint_or_n": None,
                     "n_tasks_init": 0, "init_mean": None, "init_std": None,
                     "n_tasks_final": 0, "final_mean": None, "final_std": None})
    return rows


def _cell(mean, std) -> str:
    if mean is None:
        return "--"
    if std is None:
        return f"{mean:.2f}"
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def to_latex(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcc@{}}",
        r"\toprule",
        r"Method & Initialization $\downarrow$ & Final BO $\downarrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        name = row["method"].replace("ORPT", r"\ourmethod{}") if row["method"].startswith("ORPT") else row["method"]
        lines.append(f"{name} & {_cell(row['init_mean'], row['init_std'])} & "
                     f"{_cell(row['final_mean'], row['final_std'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    return "\n".join(lines)


def generate(
    config_path: Path,
    bolt_run_dir: Path,
    orpt_run_dir: Path,
    task_set: str,
    out_dir: Path,
    target_pool_size: int | None = None,
) -> Path | None:
    cfg = peptide.load_config(str(config_path))
    pool_size = target_pool_size if target_pool_size is not None else cfg.init_size
    rows = build_rows(cfg, bolt_run_dir, orpt_run_dir, task_set, pool_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"main_results_table_{DOMAIN_SLUG}"
    csv_path = out_dir / f"{name}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[eval2.main_results_table] wrote {csv_path}")

    tex_path = out_dir / f"{name}.tex"
    tex_path.write_text(to_latex(rows) + "\n")
    print(f"[eval2.main_results_table] wrote {tex_path}")
    return csv_path
