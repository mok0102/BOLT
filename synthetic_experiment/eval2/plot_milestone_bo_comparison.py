"""Plot held-out BOLT and ORPT best scores by training milestone."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

BOLT_ROOT = Path(__file__).resolve().parents[2]
if str(BOLT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOLT_ROOT))
from synthetic_experiment.config import load_config
from data import task_curves
import style


def _boltorpt_colors(milestones: tuple[int, ...]) -> dict[int, str]:
    n = len(milestones)
    cmap = plt.get_cmap("viridis")
    return {m: mcolors.to_hex(cmap(i / (n - 1) if n > 1 else 1.0)) for i, m in enumerate(milestones)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--bolt-run', type=Path)
    parser.add_argument('--orpt-run', type=Path)
    parser.add_argument('--out-dir', type=Path, default=Path('synthetic_experiment/eval2/results/synthetic'))
    parser.add_argument('--milestones', type=int, nargs='+', default=[2, 5, 20, 40])
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not args.milestones or any(m not in cfg.milestones for m in args.milestones):
        parser.error(f'--milestones must be drawn from {cfg.milestones}')
    milestones = list(dict.fromkeys(args.milestones))
    runs = {'BOLT': args.bolt_run or cfg.run_dir, 'ORPT': args.orpt_run or cfg.run_dir}
    style.apply_rcparams()
    fig = plt.figure(figsize=(3020 / 300, 1501 / 300), dpi=300)
    ax = fig.add_axes([0.107, 0.12, 0.68, 0.86])
    colors = _boltorpt_colors(tuple(milestones))
    rows = []
    for milestone in milestones:
        for method, linestyle in (('BOLT', '--'), ('ORPT', '-')):
            curves = task_curves(runs[method], method, milestone, cfg.num_heldout_tasks,
                                 cfg.init_size, cfg.oracle_budget)
            if (curves.shape != (cfg.oracle_budget + 1, cfg.num_heldout_tasks)
                    or curves.isna().any().any()):
                raise ValueError(f'{runs[method]}: incomplete {method}-{milestone} held-out trajectories')
            mean = curves.mean(axis=1)
            std = curves.std(axis=1, ddof=1)
            sem = curves.sem(axis=1, ddof=1)
            for calls in curves.index:
                rows.append({'method': method, 'milestone': milestone, 'oracle_calls': int(calls),
                             'n_tasks': cfg.num_heldout_tasks, 'mean_best_score': float(mean.loc[calls]),
                             'std_best_score': float(std.loc[calls]), 'sem_best_score': float(sem.loc[calls])})
            ax.plot(mean.index, mean, color=colors[milestone], linestyle=linestyle,
                    linewidth=style.LINEWIDTH, label=f'{method}-{milestone}')
    style.style_axis(ax)
    ax.set_xlabel('Oracle calls after initialization')
    ax.set_ylabel('Best Branin score')
    ax.set_xlim(0, cfg.oracle_budget)
    ax.xaxis.label.set_fontsize(18)
    ax.yaxis.label.set_fontsize(18)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False,
              borderaxespad=0.0, fontsize=16, handlelength=2.0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = 'bolt_orpt_by_milestone_bo'
    pd.DataFrame(rows).to_csv(args.out_dir / f'{stem}.csv', index=False)
    path = args.out_dir / f'{stem}.png'
    with plt.rc_context({'savefig.bbox': None}):
        fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'[eval2] wrote {path}')


if __name__ == '__main__':
    main()
