"""Write ORPT horizon ablation tables from held-out Branin trajectories."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BOLT_ROOT = Path(__file__).resolve().parents[2]
if str(BOLT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOLT_ROOT))
from synthetic_experiment.config import load_config
from table_baselines import _trajectory


def _cell(mean: float, std: float, *, bold: bool = False) -> str:
    mean_text = rf'\mathbf{{{mean:.4f}}}' if bold else f'{mean:.4f}'
    return rf'${mean_text}${{\scriptsize $\pm${std:.2f}}}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-prefix', default='synthetic_branin_50train_20heldout_mi30_composite_v3_smaller_h')
    parser.add_argument('--runs-root', type=Path, default=Path('runs'))
    parser.add_argument('--out-dir', type=Path, default=Path('synthetic_experiment/eval2/results/horizon_comparison'))
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = []
    for milestone in cfg.milestones:
        for horizon in range(4):
            run_dir = args.runs_root / f'{args.run_prefix}{horizon}'
            initial, final = [], []
            for task_index, task_t in enumerate(cfg.heldout_task_values):
                frame = _trajectory(run_dir, f'ORPT-{milestone}', task_index, task_t,
                                    cfg.init_size + cfg.oracle_budget)
                scores = frame.train_y.to_numpy(dtype=float)
                if not np.isfinite(scores[:cfg.init_size + cfg.oracle_budget]).all():
                    raise ValueError(f'{run_dir}: nonfinite held-out scores at task {task_index}')
                initial.append(float(scores[:cfg.init_size].max()))
                final.append(float(scores[:cfg.init_size + cfg.oracle_budget].max()))
            rows.append({'milestone': milestone, 'horizon': horizon,
                         'n_tasks': len(initial), 'init_mean': np.mean(initial),
                         'init_std': np.std(initial, ddof=1),
                         'final_mean': np.mean(final),
                         'final_std': np.std(final, ddof=1),
                         'source_run': str(run_dir)})
    data = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / 'horizon_ablation.csv'
    data.to_csv(csv_path, index=False)

    final_milestone = cfg.milestones[-1]
    subset = data[data.milestone == final_milestone].set_index('horizon')
    best_init = int(subset.init_mean.idxmax())
    best_final = int(subset.final_mean.idxmax())
    main_lines = [
        r'\begin{wraptable}{r}{0.50\textwidth}', r'\centering', r'\footnotesize',
        rf'\caption{{ORPT horizon ablation at $m={final_milestone}$ on {cfg.num_heldout_tasks} held-out Branin tasks. Mean best score $\pm$ SD; higher is better. Final BO uses {cfg.oracle_budget} calls.}}',
        r'\label{tab:synthetic-horizon-ablation}',
        r'\begin{tabular}{@{}lcc@{}}', r'\toprule',
        r'Horizon & Initialization & Final BO \\', r'\midrule',
    ]
    for horizon, row in subset.iterrows():
        main_lines.append(
            rf'$H={horizon}$ & {_cell(row.init_mean, row.init_std, bold=horizon == best_init)} & '
            + _cell(row.final_mean, row.final_std, bold=horizon == best_final) + r' \\')
    main_lines += [r'\bottomrule', r'\end{tabular}', r'\end{wraptable}']
    main_path = args.out_dir / 'horizon_ablation_m50.tex'
    main_path.write_text('\n'.join(main_lines) + '\n')

    all_lines = [
        r'\begin{table*}[t]', r'\centering', r'\small',
        rf'\caption{{ORPT horizon ablation across training milestones on {cfg.num_heldout_tasks} held-out synthetic Branin tasks. Mean $\pm$ sample standard deviation of the best score; higher is better. Final BO uses {cfg.oracle_budget} calls after initialization. The best mean in each row is bold.}}',
        r'\label{tab:synthetic-horizon-ablation-all}',
        r'\begin{tabular}{clcccc}', r'\toprule',
        r'Milestone & Metric & $H=0$ & $H=1$ & $H=2$ & $H=3$ \\', r'\midrule',
    ]
    for milestone in cfg.milestones:
        group = data[data.milestone == milestone].set_index('horizon')
        for label, prefix in [('Initialization', 'init'), ('Final BO', 'final')]:
            best = int(group[f'{prefix}_mean'].idxmax())
            cells = [_cell(group.loc[h, f'{prefix}_mean'], group.loc[h, f'{prefix}_std'], bold=h == best)
                     for h in range(4)]
            all_lines.append(f'{milestone if prefix == "init" else ""} & {label} & '
                             + ' & '.join(cells) + r' \\')
        if milestone != cfg.milestones[-1]:
            all_lines.append(r'\midrule')
    all_lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    all_path = args.out_dir / 'horizon_ablation_all_milestones.tex'
    all_path.write_text('\n'.join(all_lines) + '\n')
    for path in (csv_path, main_path, all_path):
        print(path)


if __name__ == '__main__':
    main()
