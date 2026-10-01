"""Compare ORPT held-out BO curves across MI horizons at milestone 50."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data import task_curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=Path('runs'))
    parser.add_argument('--run-prefix', default='synthetic_branin_50train_20heldout_mi30_composite_v3_smaller_h')
    parser.add_argument('--out-dir', type=Path, default=Path('synthetic_experiment/eval2/results/horizon_comparison'))
    parser.add_argument('--milestone', type=int, default=50)
    parser.add_argument('--num-tasks', type=int, default=20)
    parser.add_argument('--init-size', type=int, default=5)
    parser.add_argument('--oracle-budget', type=int, default=50)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(12.4, 8), dpi=150)
    rows = []
    for horizon in (0, 1, 2, 3):
        run_dir = args.runs_root / f'{args.run_prefix}{horizon}'
        curves = task_curves(run_dir, 'ORPT', args.milestone, args.num_tasks,
                             args.init_size, args.oracle_budget)
        if curves.shape[1] != args.num_tasks or len(curves) != args.oracle_budget + 1 or curves.isna().any().any():
            raise ValueError(f'{run_dir}: expected {args.num_tasks} complete ORPT-{args.milestone} trajectories')
        mean = curves.mean(axis=1)
        rows.append({'method': f'ORPT H={horizon}', 'n_tasks': curves.shape[1],
                     'init_best_score': float(mean.loc[0]),
                     'final_bo_best_score': float(mean.loc[args.oracle_budget])})
        ax.plot(mean.index, mean, linewidth=2.5, label=f'ORPT H={horizon}')

    ax.set(title=f'ORPT horizon comparison at milestone {args.milestone}',
           xlabel='BO calls', ylabel='Best Branin score (higher is better)')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'horizon_comparison_m{args.milestone}'
    pd.DataFrame(rows).to_csv(args.out_dir / f'{stem}.csv', index=False)
    fig.savefig(args.out_dir / f'{stem}.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
