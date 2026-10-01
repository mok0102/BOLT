"""LaTeX tables for STBO, BOLT, and ORPT on synthetic Branin tasks."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BOLT_ROOT = Path(__file__).resolve().parents[2]
if str(BOLT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOLT_ROOT))
from synthetic_experiment.config import load_config


METHODS = ('STBO', 'BOLT', 'ORPT')


def _trajectory(run_dir: Path, arm: str, task_index: int, task_t: float, total_rows: int) -> pd.DataFrame:
    path = run_dir / 'heldout' / arm / f'task_{task_index:04d}.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if len(frame) < total_rows or not {'train_y', 'task_t'} <= set(frame):
        raise ValueError(f'{path}: expected at least {total_rows} rows and train_y/task_t columns')
    if not np.isclose(float(frame.task_t.iloc[0]), task_t):
        raise ValueError(f'{path}: task_t differs from configured heldout task {task_t}')
    return frame


def _stats(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0


def compute(cfg, run_dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for milestone in cfg.milestones:
            arm = method if method == 'STBO' else f'{method}-{milestone}'
            initial, final = [], []
            initial_scores, final_scores = [], []
            for task_index, task_t in enumerate(cfg.heldout_task_values):
                frame = _trajectory(run_dirs[method], arm, task_index, task_t, cfg.init_size + cfg.oracle_budget)
                optimum = 10 * task_t
                init_score = float(frame.train_y.iloc[:cfg.init_size].max())
                final_score = float(frame.train_y.iloc[:cfg.init_size + cfg.oracle_budget].max())
                initial_scores.append(init_score)
                final_scores.append(final_score)
                initial.append(max(-init_score - optimum, 0.0))
                final.append(max(-final_score - optimum, 0.0))
            init_mean, init_sem = _stats(initial)
            init_std = float(np.std(initial, ddof=1)) if len(initial) > 1 else float("nan")
            final_std = float(np.std(final, ddof=1)) if len(final) > 1 else float("nan")
            init_score_std = float(np.std(initial_scores, ddof=1)) if len(initial_scores) > 1 else float("nan")
            final_score_std = float(np.std(final_scores, ddof=1)) if len(final_scores) > 1 else float("nan")
            final_mean, final_sem = _stats(final)
            init_score_mean, init_score_sem = _stats(initial_scores)
            final_score_mean, final_score_sem = _stats(final_scores)
            rows.append({'method': method, 'milestone': milestone, 'n_tasks': len(initial),
                         'source_run': str(run_dirs[method]),
                         'init_mean_regret': init_mean, 'init_std': init_std, 'init_sem': init_sem,
                         'final_mean_regret': final_mean, 'final_std': final_std, 'final_sem': final_sem,
                         'init_mean_score': init_score_mean, 'init_score_std': init_score_std, 'init_score_sem': init_score_sem,
                         'final_mean_score': final_score_mean, 'final_score_std': final_score_std, 'final_score_sem': final_score_sem})
    return pd.DataFrame(rows)


def _cell(mean: float, sem: float) -> str:
    return f'{mean:.3f} $\\pm$ {sem:.3f}'


def write_tables(data: pd.DataFrame, out_dir: Path, orpt_horizon: int | None = None) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_dir / 'stbo_bolt_orpt_regret.csv', index=False)
    milestones = sorted(data.milestone.unique())
    labels = {'ORPT': rf'ORPT ($H={orpt_horizon}$)' if orpt_horizon is not None else 'ORPT'}
    horizon_note = f' ORPT uses $H={orpt_horizon}$.' if orpt_horizon is not None else ''
    by_milestone = [
        r'\begin{table}[t]', r'\centering',
        rf'\caption{{Held-out simple regret after 50 BO calls on the synthetic Branin tasks (mean $\pm$ standard error over 20 tasks; lower is better). STBO is independent of the training milestone.{horizon_note}}}',
        r'\label{tab:synthetic-final-regret}',
        r'\begin{tabular}{l' + 'c' * len(milestones) + '}',
        r'\toprule', 'Method & ' + ' & '.join(f'$m={m}$' for m in milestones) + r' \\', r'\midrule',
    ]
    for method in METHODS:
        subset = data[data.method == method].set_index('milestone')
        by_milestone.append(labels.get(method, method) + ' & ' + ' & '.join(
            _cell(subset.loc[m, 'final_mean_regret'], subset.loc[m, 'final_sem']) for m in milestones) + r' \\')
    by_milestone += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    milestone_path = out_dir / 'stbo_bolt_orpt_by_milestone.tex'
    milestone_path.write_text('\n'.join(by_milestone) + '\n')

    final_milestone = milestones[-1]
    subset = data[data.milestone == final_milestone].set_index('method')
    final_table = [
        r'\begin{table}[t]', r'\centering',
        rf'\caption{{Held-out simple regret on synthetic Branin at milestone $m={final_milestone}$ (mean $\pm$ standard error over 20 tasks; lower is better).{horizon_note}}}',
        r'\label{tab:synthetic-initial-final-regret}', r'\begin{tabular}{lcc}', r'\toprule',
        r'Method & Initial pool & After 50 BO calls \\', r'\midrule',
    ]
    for method in METHODS:
        row = subset.loc[method]
        final_table.append(f"{labels.get(method, method)} & {_cell(row.init_mean_regret, row.init_sem)} & {_cell(row.final_mean_regret, row.final_sem)} " + r'\\')
    final_table += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    final_path = out_dir / 'stbo_bolt_orpt_m50.tex'
    final_path.write_text('\n'.join(final_table) + '\n')
    return milestone_path, final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stbo-run', type=Path, required=True)
    parser.add_argument('--bolt-run', type=Path, required=True)
    parser.add_argument('--orpt-run', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--orpt-horizon', type=int, help='H used by the ORPT run, for the table label')
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = compute(cfg, {'STBO': args.stbo_run, 'BOLT': args.bolt_run, 'ORPT': args.orpt_run})
    for path in write_tables(data, args.out_dir, args.orpt_horizon):
        print(path)


if __name__ == '__main__':
    main()
