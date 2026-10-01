"""Mean heldout best score versus BO calls after initialization."""
from pathlib import Path
import matplotlib.pyplot as plt
import style
from data import task_curves


PAPER_COLORS = {
    'BOLT-50': '#3C5488', 'ORPT-50': '#E64B35',
    'POGPE-5': '#E6E0D7', 'POGPE-10': '#C9BCAE', 'POGPE-20': '#A9927B',
    'POGPE-50': '#6E5846',
    'SGPE-5': '#D8EFEA', 'SGPE-10': '#A4DCCF', 'SGPE-20': '#71C8B4',
    'SGPE-50': '#267C6A',
    'MTBO': '#4DBBD5', 'OptFormer': '#7E6148', 'STBO': '#00A087',
}


def _arm_milestone(label: str, default_milestone: int) -> tuple[str, int]:
    arm, sep, number = label.rpartition('-')
    return (arm, int(number)) if sep and number.isdigit() else (label, default_milestone)


def generate(cfg, milestone: int, run_dirs: dict[str, Path], out_dir: Path, *, inset: bool = False) -> Path | None:
    style.apply_rcparams()
    series = []
    for label, run_dir in run_dirs.items():
        arm, arm_milestone = _arm_milestone(label, milestone)
        curves = task_curves(run_dir, arm, arm_milestone, cfg.num_heldout_tasks, cfg.init_size, cfg.oracle_budget)
        if curves.empty:
            print(f'[eval2.main_bo] {label}: no heldout trajectories, skipping')
            continue
        mean = curves.mean(axis=1)
        # Use the fixed task's known optimum to remove variation in score
        # level due to t before estimating uncertainty across tasks.
        adjusted = curves.copy()
        for task_index in adjusted.columns:
            adjusted[task_index] += 10 * cfg.heldout_task_values[task_index]
        half_width = adjusted.sem(axis=1)
        series.append((label, arm, curves.index, mean, half_width))
        print(f'[eval2.main_bo] {label}: {curves.shape[1]} tasks')
    if not series:
        return None

    if inset:
        fig = plt.figure(figsize=(3057 / 300, 1501 / 300), dpi=300)
        ax = fig.add_axes([0.11, 0.13, 0.666, 0.78])
        detail = fig.add_axes([0.53, 0.305, 0.215, 0.32], facecolor='white', zorder=5)
    else:
        fig, (ax, detail) = plt.subplots(
            1, 2, figsize=(10.8, 3.25), gridspec_kw={'width_ratios': [1.15, 1]}
        )
    markers = {
        'BOLT': 'o', 'ORPT': '*', 'STBO': '^', 'MTBO': 'v',
        'POGPE': 'D', 'SGPE': 's', 'OptFormer': 'P', 'LLAMBO': 'X',
    }
    lines = []
    for label, arm, xs, ys, half_width in series:
        color = PAPER_COLORS.get(label, style.arm_color(arm))
        kwargs = dict(color=color, linewidth=style.LINEWIDTH, label=label)
        if not inset:
            kwargs.update(marker=markers[arm], markersize=4.2, markevery=5, markeredgewidth=0.35)
        line, = ax.plot(xs, ys, **kwargs)
        detail.plot(xs, ys, **kwargs)
        if not inset:
            for panel in (ax, detail):
                panel.fill_between(xs, ys - half_width, ys + half_width,
                                   color=color, alpha=0.10, linewidth=0)
        lines.append(line)

    style.style_axis(ax)
    ax.set_title('' if inset else 'All methods')
    ax.set_xlabel('Oracle calls after initialization')
    ax.set_ylabel('Best objective found (↑)')
    if inset:
        ax.title.set_fontsize(18)
        ax.xaxis.label.set_fontsize(18)
        ax.yaxis.label.set_fontsize(18)
        ax.tick_params(axis='both', labelsize=13)

    style.style_axis(detail)
    detail.set_title('Score detail', fontsize=12 if inset else None)
    if inset:
        detail.tick_params(axis='both', labelsize=10, length=2)
        for spine in detail.spines.values():
            spine.set_visible(True)
            spine.set_color('#bcbcbc')
    else:
        detail.set_xlabel('Oracle calls after initialization')
        detail.set_ylabel('Score')
    detail.set_ylim(-4.783, -4.752)
    detail.ticklabel_format(axis='y', style='plain', useOffset=False)
    if inset:
        fig.legend(handles=lines, loc='center left', bbox_to_anchor=(0.78, 0.5), frameon=False, fontsize=14)
    else:
        fig.subplots_adjust(left=0.075, right=0.84, bottom=0.18, top=0.89, wspace=0.18)
        fig.legend(handles=lines, loc='center left', bbox_to_anchor=(0.855, 0.5), frameon=False)
    name = 'main_bo_synthetic_new' if inset else 'main_bo_synthetic'
    if inset:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f'{name}.png'
        with plt.rc_context({'savefig.bbox': None}):
            fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f'[eval2] wrote {path}')
        return path
    return style.savefig(fig, out_dir, name)
