"""Synthetic best-of-k negative Branin scores with final-BO baseline references."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BOLT_ROOT = Path(__file__).resolve().parents[2]
if str(BOLT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOLT_ROOT))
from synthetic_experiment.config import load_config
from data import init_curves, task_curves
import style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--milestone', type=int, default=50)
    parser.add_argument('--bolt-milestone', type=int)
    parser.add_argument('--orpt-milestone', type=int)
    parser.add_argument('--bolt-run', type=Path)
    parser.add_argument('--orpt-run', type=Path)
    parser.add_argument('--reference-run', action='append', default=[], metavar='ARM=PATH')
    parser.add_argument('--out-dir', type=Path, default=Path('synthetic_experiment/eval2/results/synthetic'))
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.milestone not in cfg.milestones:
        parser.error(f'--milestone must be in {cfg.milestones}')
    bolt_milestone = args.bolt_milestone or args.milestone
    if bolt_milestone not in cfg.milestones:
        parser.error(f'--bolt-milestone must be in {cfg.milestones}')
    orpt_milestone = args.orpt_milestone or args.milestone
    if orpt_milestone not in cfg.milestones:
        parser.error(f'--orpt-milestone must be in {cfg.milestones}')
    runs = {'BOLT': args.bolt_run or cfg.run_dir, 'ORPT': args.orpt_run or cfg.run_dir}
    references = {}
    for item in args.reference_run:
        arm, sep, path = item.partition('=')
        if not sep or arm in runs or arm in references or not path:
            parser.error(f'expected unique baseline ARM=PATH, got {item!r}')
        references[arm] = Path(path)

    style.apply_rcparams()
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(7.5, 3.4),
                                      gridspec_kw={'height_ratios': [4.4, 0.55], 'hspace': 0.14})
    rows = []
    handles = []
    curve_values = []
    upper_reference_values = []
    lower_reference_values = []
    orpt_detail = None
    detail_references = []

    def score_curves(arm: str, run_dir: Path, milestone: int) -> pd.DataFrame:
        curves = init_curves(run_dir, arm, milestone, cfg.num_heldout_tasks, cfg.init_size)
        if curves.shape != (cfg.init_size, cfg.num_heldout_tasks) or curves.isna().any().any():
            raise ValueError(f'{run_dir}: incomplete {arm}-{milestone} initial proposals')
        # Stored train_y is the negative Branin objective; higher is better.
        return curves

    for arm, run_dir in runs.items():
        milestone = orpt_milestone if arm == 'ORPT' else bolt_milestone
        curves = score_curves(arm, run_dir, milestone)
        means = curves.mean(axis=1)
        if arm == 'ORPT':
            orpt_detail = (curves.index, means)
        for k, values in curves.iterrows():
            rows.append({'arm': arm, 'kind': 'best_of_k', 'milestone': milestone,
                         'k': int(k), 'n_tasks': len(values),
                         'mean_branin_score': float(means.loc[k]),
                         'std_branin_score': float(values.std(ddof=1)),
                         'source_run': str(run_dir)})
        curve_values.extend(float(value) for value in means)
        line, = top.plot(curves.index, means, marker='o', linewidth=style.LINEWIDTH,
                            color=style.arm_color(arm), label=f'{arm}-{milestone}')
        handles.append(line)

    for arm, run_dir in references.items():
        curves = task_curves(run_dir, arm, args.milestone, cfg.num_heldout_tasks,
                             cfg.init_size, cfg.oracle_budget)
        if curves.shape != (cfg.oracle_budget + 1, cfg.num_heldout_tasks) or curves.isna().any().any():
            raise ValueError(f'{run_dir}: incomplete {arm}-{args.milestone} final BO trajectories')
        values = curves.loc[cfg.oracle_budget]
        mean = float(values.mean())
        rows.append({'arm': arm, 'kind': 'final_bo_reference', 'milestone': args.milestone,
                     'k': pd.NA, 'n_tasks': len(values),
                     'mean_branin_score': mean,
                     'std_branin_score': float(values.std(ddof=1)),
                     'source_run': str(run_dir)})
        lower = mean < min(curve_values) - 0.5
        (lower_reference_values if lower else upper_reference_values).append(mean)
        ax = bottom if lower else top
        if not lower:
            detail_references.append((arm, mean))
        line = ax.axhline(mean, linestyle=':', linewidth=style.LINEWIDTH,
                          color=style.arm_color(arm), label=arm)
        handles.append(line)

    for ax in (top, bottom):
        style.style_axis(ax)
        ax.set_xlim(1, cfg.init_size)
    low, high = min(curve_values + upper_reference_values), max(curve_values + upper_reference_values)
    top.set_ylim(low - 0.01, high + 0.01)
    if lower_reference_values:
        low, high = min(lower_reference_values), max(lower_reference_values)
        bottom.set_ylim(low - 0.045, high + 0.045)
    if orpt_detail is not None:
        detail = top.inset_axes([0.53, 0.13, 0.43, 0.33], facecolor='white', zorder=5)
        detail.plot(*orpt_detail, marker='o', linewidth=style.LINEWIDTH,
                    markersize=style.MARKERSIZE, color=style.arm_color('ORPT'))
        for arm, mean in detail_references:
            detail.axhline(mean, linestyle=':', linewidth=style.LINEWIDTH,
                           color=style.arm_color(arm))
        detail.set_xlim(1, cfg.init_size)
        detail.set_ylim(-4.768, -4.752)
        detail.set_xticks(range(1, cfg.init_size + 1))
        detail.set_yticks((-4.765, -4.760, -4.755))
        detail.set_title('ORPT detail', fontsize=8)
        detail.tick_params(labelsize=7, length=2)
        detail.grid(True, linewidth=0.5, color=style.GRID_COLOR)
        for spine in detail.spines.values():
            spine.set_visible(True)
            spine.set_color(style.MUTED_TEXT)
    top.spines['bottom'].set_visible(False)
    bottom.spines['top'].set_visible(False)
    top.tick_params(axis='x', bottom=False, labelbottom=False)
    mark = dict(color=style.MUTED_TEXT, clip_on=False, linewidth=1.2)
    for x in (-0.012, 0.5, 1.012):
        top.plot((x - 0.012, x + 0.012), (-0.035, 0.035), transform=top.transAxes, **mark)
        bottom.plot((x - 0.012, x + 0.012), (0.965, 1.035), transform=bottom.transAxes, **mark)
    bottom.set_xlabel('Number of proposal samples (k)')
    bottom.set_xticks(range(1, cfg.init_size + 1))
    fig.supylabel('Best Branin score', x=0.08)
    style.place_legend(top, handles, [line.get_label() for line in handles])
    fig.subplots_adjust(left=0.17, right=0.72, bottom=0.15, top=0.96)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = 'fewshot_synthetic_reference'
    pd.DataFrame(rows).to_csv(args.out_dir / f'{stem}.csv', index=False)
    style.savefig(fig, args.out_dir, stem)


if __name__ == '__main__':
    main()
