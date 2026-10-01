"""Plot synthetic STBO/BOLT/ORPT regret and best scores from table CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd

COLORS = {'STBO': '#555555', 'BOLT': '#3C5488', 'ORPT': '#DC0000'}
MARKERS = {'STBO': 's', 'BOLT': 'o', 'ORPT': '^'}


def plot(csv_path: Path, out_dir: Path, orpt_horizon: int | None = None) -> tuple[Path, Path]:
    data = pd.read_csv(csv_path)
    required = {'method', 'milestone', 'n_tasks', 'init_mean_regret', 'init_sem', 'final_mean_regret', 'final_sem',
                'init_mean_score', 'init_score_sem', 'final_mean_score', 'final_score_sem'}
    if not required <= set(data):
        raise ValueError(f'{csv_path}: missing {sorted(required - set(data))}')
    if set(data.method) != set(COLORS) or data.n_tasks.nunique() != 1:
        raise ValueError('expected STBO, BOLT, ORPT with the same heldout task count')
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 300, 'figure.dpi': 150})
    orpt_label = f'ORPT ($H={orpt_horizon}$)' if orpt_horizon is not None else 'ORPT'
    labels = {'STBO': 'STBO', 'BOLT': 'BOLT', 'ORPT': orpt_label}

    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    for method in COLORS:
        rows = data[data.method == method].sort_values('milestone')
        ax.errorbar(rows.milestone, rows.final_mean_regret, yerr=rows.final_sem,
                    color=COLORS[method], marker=MARKERS[method], markersize=5,
                    linewidth=1.8, ecolor=to_rgba(COLORS[method], 0.28),
                    elinewidth=0.8, capsize=1.5, capthick=0.8, label=labels[method])
    ax.set_xticks(sorted(data.milestone.unique()))
    ax.set_xlabel('Completed training tasks (milestone)')
    ax.set_ylabel('Final simple regret after 50 BO calls $\\downarrow$')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', color='#e1e0d9', linewidth=0.7)
    ax.legend(frameon=False, ncol=3, loc='upper right')
    path_milestones = out_dir / 'stbo_bolt_orpt_by_milestone.png'
    fig.savefig(path_milestones)
    fig.savefig(path_milestones.with_suffix('.pdf'))
    plt.close(fig)

    milestone = int(data.milestone.max())
    final = data[data.milestone == milestone].set_index('method')
    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    x = np.arange(2)
    offsets = {'STBO': -0.22, 'BOLT': 0.0, 'ORPT': 0.22}
    for method in COLORS:
        row = final.loc[method]
        ax.errorbar(x + offsets[method], [row.init_mean_regret, row.final_mean_regret],
                    yerr=[row.init_sem, row.final_sem], color=COLORS[method],
                    marker=MARKERS[method], markersize=6, linewidth=1.8,
                    ecolor=to_rgba(COLORS[method], 0.28), elinewidth=0.8,
                    capsize=1.5, capthick=0.8, label=labels[method])
    ax.set_xticks(x, ['Initial pool', 'After 50 BO calls'])
    ax.set_yscale('log')
    ax.set_ylabel('Simple regret $\\downarrow$ (log scale)')
    ax.set_title(f'Training milestone $m={milestone}$', fontsize=11)
    ax.grid(axis='y', color='#e1e0d9', linewidth=0.7)
    ax.legend(frameon=False, ncol=3, loc='upper right')
    path_m50 = out_dir / 'stbo_bolt_orpt_m50.png'
    fig.savefig(path_m50)
    fig.savefig(path_m50.with_suffix('.pdf'))
    plt.close(fig)

    score_paths = []
    for metric, ylabel, filename in (
        ('final_mean_score',
         'Best score after 50 BO calls $\\uparrow$', 'stbo_bolt_orpt_final_score'),
        ('init_mean_score',
         'Best initialization score $\\uparrow$', 'stbo_bolt_orpt_initialization_best'),
    ):
        if metric == 'init_mean_score':
            fig, (upper, lower) = plt.subplots(
                2, 1, sharex=True, figsize=(6.4, 4.4),
                gridspec_kw={'height_ratios': [2.2, 1.0], 'hspace': 0.08},
            )
            handles = []
            for method in COLORS:
                rows = data[data.method == method].sort_values('milestone')
                ax = lower if method == 'STBO' else upper
                handle, = ax.plot(rows.milestone, rows[metric],
                                  color=COLORS[method], marker=MARKERS[method],
                                  markersize=5, linewidth=1.8, label=labels[method])
                handles.append(handle)
            comparison = data[data.method != 'STBO'][metric]
            span = float(comparison.max() - comparison.min())
            pad = max(span * 0.15, 0.005)
            upper.set_ylim(float(comparison.min()) - pad, float(comparison.max()) + pad)
            baseline = data[data.method == 'STBO'][metric]
            lower.set_ylim(float(baseline.min()) - 0.25, float(baseline.max()) + 0.25)
            upper.spines['bottom'].set_visible(False)
            lower.spines['top'].set_visible(False)
            upper.tick_params(bottom=False, labelbottom=False)
            lower.tick_params(top=False)
            for ax in (upper, lower):
                ax.set_xticks(sorted(data.milestone.unique()))
                ax.grid(axis='y', color='#e1e0d9', linewidth=0.7)
            lower.set_xlabel('Completed training tasks (milestone)')
            fig.supylabel(ylabel, x=0.02)
            upper.legend(handles=handles, frameon=False, ncol=3, loc='upper right')
            # Small diagonal marks make the discontinuity in the shared y-axis explicit.
            d = 0.008
            for ax, y in ((upper, 0), (lower, 1)):
                for x in (0, 1):
                    ax.plot((x - d, x + d), (y - d, y + d), transform=ax.transAxes,
                            color='#555555', linewidth=0.9, clip_on=False)
            fig.subplots_adjust(left=0.17, right=0.98, bottom=0.13, top=0.96)
            path = out_dir / (filename + '.png')
            fig.savefig(path)
            fig.savefig(path.with_suffix('.pdf'))
            plt.close(fig)
            score_paths.append(path)
            continue
        fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
        for method in COLORS:
            rows = data[data.method == method].sort_values('milestone')
            ax.plot(rows.milestone, rows[metric],
                    color=COLORS[method], marker=MARKERS[method], markersize=5,
                    linewidth=1.8, label=labels[method])
        ax.set_xticks(sorted(data.milestone.unique()))
        ax.set_xlabel('Completed training tasks (milestone)')
        ax.set_ylabel(ylabel)
        if metric == 'final_mean_score':
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
        ax.grid(axis='y', color='#e1e0d9', linewidth=0.7)
        ax.legend(frameon=False, ncol=3, loc='lower right')
        path = out_dir / (filename + '.png')
        fig.savefig(path)
        fig.savefig(path.with_suffix('.pdf'))
        plt.close(fig)
        score_paths.append(path)
    return path_milestones, path_m50, *score_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--orpt-horizon', type=int)
    args = parser.parse_args()
    for path in plot(args.csv, args.out_dir, args.orpt_horizon):
        print(path)


if __name__ == '__main__':
    main()
