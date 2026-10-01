"""Heldout initialization and final BO score versus training milestone."""
from pathlib import Path
import matplotlib.pyplot as plt
import style
from data import task_curves


def _limits(values: list[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    pad = max((high - low) * 0.12, 0.012 if low < -9 else 0.002)
    return low - pad, high + pad


def generate(cfg, run_dirs: dict[str, Path], out_dir: Path, final: bool) -> Path | None:
    style.apply_rcparams()
    series = []
    for arm, run_dir in run_dirs.items():
        if not final and arm not in {"BOLT", "ORPT", "OptFormer"}:
            continue
        xs, ys = [], []
        for milestone in cfg.milestones:
            curves = task_curves(run_dir, arm, milestone, cfg.num_heldout_tasks, cfg.init_size, cfg.oracle_budget)
            if curves.empty:
                continue
            target = cfg.oracle_budget if final else 0
            if target not in curves.index or curves.loc[target].isna().all():
                continue
            xs.append(milestone)
            ys.append(float(curves.loc[target].mean()))
        if xs:
            series.append((arm, xs, ys))
    if not series:
        return None

    # Separate widely spaced score bands so every method remains legible.
    upper = [y for _, _, ys in series for y in ys if y > -4.9]
    lower = [y for _, _, ys in series for y in ys if y <= -4.9]
    if upper and lower:
        fig, (top, bottom) = plt.subplots(
            2, 1, sharex=True, figsize=(style.FIGSIZE[0], 3.45),
            gridspec_kw={'height_ratios': [2.2, 1], 'hspace': 0.08},
        )
        axes = (top, bottom)
        top.set_ylim(*_limits(upper))
        bottom.set_ylim(*_limits(lower))
        top.spines['bottom'].set_visible(False)
        bottom.spines['top'].set_visible(False)
        top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        mark = dict(color=style.MUTED_TEXT, clip_on=False, linewidth=0.8)
        for x in (-0.012, 1.012):
            top.plot((x - 0.008, x + 0.008), (-0.012, 0.012), transform=top.transAxes, **mark)
            bottom.plot((x - 0.008, x + 0.008), (0.988, 1.012), transform=bottom.transAxes, **mark)
    else:
        fig, top = plt.subplots(figsize=style.FIGSIZE)
        axes = (top,)

    for arm, xs, ys in series:
        for ax in axes:
            ax.plot(xs, ys, color=style.arm_color(arm), linewidth=style.LINEWIDTH,
                    linestyle='--' if arm in {'STBO', 'LLAMBO'} or final and arm == 'OptFormer' else '-',
                    marker=None if arm in {'STBO', 'LLAMBO'} else 'o',
                    markersize=style.MARKERSIZE, label=arm)
    for ax in axes:
        style.style_axis(ax)
    axes[-1].set_xlabel('Completed training tasks')
    fig.supylabel(('Final BO' if final else 'Initialization') + ' best Branin score (higher is better)', fontsize=9)
    axes[-1].set_xticks(cfg.milestones)
    style.place_legend(top)
    return style.savefig(fig, out_dir, f"scaling_synthetic_{'finalbo' if final else 'init'}")
