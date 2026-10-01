"""Mean best-of-k initial proposal score for k=1,...,init_size."""
from pathlib import Path
import matplotlib.pyplot as plt
import style
from data import init_curves


def generate(cfg, milestone: int, run_dirs: dict[str, Path], out_dir: Path) -> Path | None:
    style.apply_rcparams()
    fig, ax = plt.subplots(figsize=style.FIGSIZE)
    plotted = False
    for arm, run_dir in run_dirs.items():
        curves = init_curves(run_dir, arm, milestone, cfg.num_heldout_tasks, cfg.init_size)
        if curves.empty:
            continue
        ax.plot(curves.index, curves.mean(axis=1), color=style.arm_color(arm), linewidth=style.LINEWIDTH,
                linestyle='--' if arm in {'STBO', 'LLAMBO'} else '-',
                marker=None if arm in {'STBO', 'LLAMBO'} else 'o',
                markersize=style.MARKERSIZE, label=arm)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    style.style_axis(ax)
    ax.set_xlabel('Initial proposal count (k)')
    ax.set_ylabel('Best-of-k Branin score (higher is better)')
    ax.set_xticks(range(1, cfg.init_size + 1))
    style.place_legend(ax)
    return style.savefig(fig, out_dir, 'fewshot_synthetic')
