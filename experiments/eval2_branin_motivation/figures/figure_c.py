"""Figure C: what should a learned initializer optimize?

At T=50 (milestone 50), plots downstream BO trajectories for every
initializer variant and produces a compact result table, per
impl_plan/motivational_exp.txt's exact spec:

  motivation_C_initializer_objective.pdf/png
  motivation_C_table.csv / .tex  (Method, Initialization regret,
                                  Regret @ 5, Regret @ 10, Regret @ 50)

Exact x/y-axis strings (reproduced verbatim, not paraphrased):
  x-axis: "Oracle calls after initialization"
  y-axis: "Simple regret to global optimum ↓"

Legend labels are the doc's own preferred text, via LEGEND_LABEL below:
  Random / Prior-best reuse / Context-to-optimum regression /
  BOLT (Top-K SFT) / H=0 preference / ORPT (H=1)

"BOLT (Top-K SFT)", not "Top-K SFT": this package's BOLT arm IS
synthetic_experiment's own real BOLT baseline (trajectory_chain.py's
build_sft_data + train_milestone, unmodified), not an approximate
reproduction -- the doc is explicit that only an exact implementation may
use the "BOLT (...)" form.

"H=0 preference" and "ORPT (H=1)" are literally the same arm name (ORPT) at
the same milestone, from two DIFFERENT run_dirs (configs/h0_gpu45.yaml vs.
h1_gpu23.yaml -- mi_bo_steps=0 vs. 1 is the only difference between the two
training runs that produced them). H=3 (configs/h3_gpu67.yaml) is plotted as
a secondary/optional overlay per the doc's "if existing H=2 or H=3 ... but
do not make the main plot unreadable" -- off by default.

generate_trajectory() also overlays a zoomed inset (b in INSET_B_RANGE) in
the main panel's empty upper-right region, since BOLT/ORPT-H0/ORPT-H1/ORPT-H3
(and Prior-best reuse) are otherwise indistinguishable once converged near
zero on the main panel's full-range y-axis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[1]
BOLT_ROOT = PKG_ROOT.parents[1]
for _p in (BOLT_ROOT, PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import style  # noqa: E402
from core.regret import full_trajectory_table, regret_table, trajectory_path  # noqa: E402
from synthetic_experiment.branin import BRANIN_BOUNDS, DEFAULT_PARAMETERS  # noqa: E402
from synthetic_experiment.config import load_config  # noqa: E402

MILESTONE = 50
CHECKPOINTS = (0, 5, 10, 50)

# Inset zoom region for generate_trajectory: BOLT/ORPT-H0/ORPT-H1/ORPT-H3
# (and Prior-best reuse) are visually indistinguishable in the main panel
# once they've all decayed near zero -- b in [30,40] sits in a region of the
# main axes that every method's curve has already passed through (Random/
# ContextRegression are also near-converged there), so it doubles as empty
# plotting area for the inset itself.
INSET_B_RANGE = (30, 40)
INSET_YLIM = (0.0, 0.045)
INSET_BOUNDS = (0.40, 0.38, 0.56, 0.58)  # (x0, y0, width, height), axes-fraction

# generate_init_pools: which held-out tasks to show (index into
# heldout_tasks, fixed/deterministic -- first, middle, last of the 20 tasks,
# not selected by any downstream result).
TASK_INDICES_FOR_INIT_POOLS = (0, 10, 19)
# Contour window: each panel's colormap spans [local min, local min + WINDOW]
# of the RAW (minimize-convention) Branin value. An input-space transform
# relabels WHERE the canonical landscape sits but never changes its values,
# so every task shares the exact same well-to-pass height along the valley,
# 2*s*(1-t) at the family's fixed canonical t -- unlike the retired scalar-t
# family, this is now a single constant that is correct for every task, not
# a per-task-varying quantity.
CONTOUR_WINDOW = 2.0 * DEFAULT_PARAMETERS.s * (1.0 - DEFAULT_PARAMETERS.t)

# (method_key, arm, uses_h3_overlay_only) -- method_key doubles as the
# style.ARM_COLOR lookup key (see style.py's Experiment-C purple-family
# addition for Random/PriorBestReuse/ContextRegression; BOLT/ORPT reuse
# their existing Figure-A colors).
MAIN_METHODS = ("Random", "PriorBestReuse", "ContextRegression", "BOLT", "ORPT-H0", "ORPT-H1")
SECONDARY_METHODS = ("ORPT-H3",)

LEGEND_LABEL = {
    "Random": "Random",
    "PriorBestReuse": "Prior-best reuse",
    "ContextRegression": "Context-to-optimum regression",
    "BOLT": "BOLT (Top-K SFT)",
    "ORPT-H0": "H=0 preference",
    "ORPT-H1": "ORPT (H=1)",
    "ORPT-H3": "ORPT (H=3)",
}


def _resolve(method: str, baselines_cfg, h0_cfg, h1_cfg, h3_cfg):
    """(run_dir, arm, milestone) for a Figure-C method key."""
    if method in ("Random", "PriorBestReuse", "ContextRegression"):
        return baselines_cfg.run_dir, method, None
    if method == "BOLT":
        return baselines_cfg.run_dir, "BOLT", MILESTONE
    if method == "ORPT-H0":
        return h0_cfg.run_dir, "ORPT", MILESTONE
    if method == "ORPT-H1":
        return h1_cfg.run_dir, "ORPT", MILESTONE
    if method == "ORPT-H3":
        if h3_cfg is None:
            raise ValueError("ORPT-H3 requested but no h3_config given")
        return h3_cfg.run_dir, "ORPT", MILESTONE
    raise ValueError(f"unknown Figure C method {method!r}")


def generate_trajectory(
    baselines_config: str, h0_config: str, h1_config: str, out_dir: Path,
    h3_config: str | None = None, include_h3: bool = False,
) -> tuple[Path, Path]:
    baselines_cfg = load_config(baselines_config)
    h0_cfg = load_config(h0_config)
    h1_cfg = load_config(h1_config)
    h3_cfg = load_config(h3_config) if h3_config else None
    for cfg, name in ((h0_cfg, "h0"), (h1_cfg, "h1"), *([(h3_cfg, "h3")] if h3_cfg else [])):
        if cfg.manifest.token != baselines_cfg.manifest.token:
            raise ValueError(f"{name} config's task manifest differs from the baselines config")

    methods = list(MAIN_METHODS) + (list(SECONDARY_METHODS) if include_h3 else [])
    rows = []
    for method in methods:
        run_dir, arm, milestone = _resolve(method, baselines_cfg, h0_cfg, h1_cfg, h3_cfg)
        t = full_trajectory_table(run_dir, arm, milestone, baselines_cfg.heldout_tasks,
                                  baselines_cfg.init_size, baselines_cfg.oracle_budget)
        t.insert(0, "method", method)
        rows.append(t)
    data = pd.concat(rows, ignore_index=True)

    style.apply_rcparams()
    fig, ax = plt.subplots(figsize=style.FIGSIZE)
    for method in methods:
        series = data[data.method == method].groupby("b").simple_regret.mean()
        # style.ABLATION_LINESTYLE's keys (BOLT, ORPT-H0, ORPT-H1, ORPT-H3) are
        # exactly this module's method keys already -- no translation needed.
        # Random/PriorBestReuse/ContextRegression aren't in that dict and fall
        # back to a plain solid line via .get's default.
        linestyle = style.ABLATION_LINESTYLE.get(method, "-")
        ax.plot(series.index, series.values, color=style.arm_color(method),
                linestyle=linestyle, linewidth=style.LINEWIDTH, label=LEGEND_LABEL[method])
    style.style_axis(ax)
    ax.set_xlabel("Oracle calls after initialization")
    ax.set_ylabel("Simple regret to global optimum ↓")
    ax.set_xlim(0, baselines_cfg.oracle_budget)
    style.place_legend(ax)
    fig.tight_layout()

    # Inset added AFTER tight_layout: inset_axes isn't gridspec-backed, and
    # calling tight_layout() with it already present either warns ("not
    # compatible with tight_layout") or tries to allocate it grid space --
    # adding it once the main axes' position is final avoids both.
    axins = ax.inset_axes(INSET_BOUNDS)
    for method in methods:
        series = data[data.method == method].groupby("b").simple_regret.mean()
        linestyle = style.ABLATION_LINESTYLE.get(method, "-")
        axins.plot(series.index, series.values, color=style.arm_color(method),
                   linestyle=linestyle, linewidth=style.LINEWIDTH * 0.85)
    axins.set_xlim(*INSET_B_RANGE)
    axins.set_ylim(*INSET_YLIM)
    axins.set_xticks([INSET_B_RANGE[0], sum(INSET_B_RANGE) // 2, INSET_B_RANGE[1]])
    axins.set_yticks([INSET_YLIM[0], INSET_YLIM[1] / 2, INSET_YLIM[1]])
    axins.tick_params(labelsize=6, length=2, pad=1.5)
    axins.grid(True, linewidth=0.4, color=style.GRID_COLOR)
    for spine in axins.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color(style.MUTED_TEXT)
    ax.indicate_inset_zoom(axins, edgecolor=style.MUTED_TEXT, linewidth=0.7, alpha=0.7)

    out_dir.mkdir(parents=True, exist_ok=True)
    style.save_csv(data, out_dir, "motivation_C_initializer_objective")
    return style.savefig(fig, out_dir, "motivation_C_initializer_objective")


def generate_table(
    baselines_config: str, h0_config: str, h1_config: str, out_dir: Path,
    h3_config: str | None = None, include_h3: bool = False,
) -> tuple[Path, Path]:
    baselines_cfg = load_config(baselines_config)
    h0_cfg = load_config(h0_config)
    h1_cfg = load_config(h1_config)
    h3_cfg = load_config(h3_config) if h3_config else None

    methods = list(MAIN_METHODS) + (list(SECONDARY_METHODS) if include_h3 else [])
    rows = []
    for method in methods:
        run_dir, arm, milestone = _resolve(method, baselines_cfg, h0_cfg, h1_cfg, h3_cfg)
        t = regret_table(run_dir, arm, milestone, baselines_cfg.heldout_tasks,
                         baselines_cfg.init_size, baselines_cfg.oracle_budget, list(CHECKPOINTS))
        agg = t.groupby("b").simple_regret.mean()
        rows.append({
            "Method": LEGEND_LABEL[method],
            "Initialization regret": agg.loc[0],
            "Regret @ 5": agg.loc[5],
            "Regret @ 10": agg.loc[10],
            "Regret @ 50": agg.loc[50],
        })
    table = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = style.save_csv(table, out_dir, "motivation_C_table")

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Experiment C: initializer training signal, simple regret to the "
        r"verified global optimum (mean over 20 held-out tasks; lower is better) at "
        r"milestone $T=50$.}",
        r"\label{tab:motivation-c}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"Method & Init. & @5 & @10 & @50 \\", r"\midrule",
    ]
    for row in table.itertuples():
        lines.append(
            f"{row.Method} & {row._2:.3f} & {row._3:.3f} & {row._4:.3f} & {row._5:.3f} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = out_dir / "motivation_C_table.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"[eval2_branin_motivation] wrote {tex_path}")
    return csv_path, tex_path


def _init_pool(run_dir: Path, arm: str, milestone: int | None, task, init_size: int) -> np.ndarray:
    """The first `init_size` (x1, x2) points evaluated for one held-out task
    -- rows 0..init_size-1 of its trajectory CSV, before any BO acquisition."""
    frame = pd.read_csv(trajectory_path(run_dir, arm, milestone, task)).head(init_size)
    return np.array([json.loads(v) for v in frame.train_x], dtype=float)


def generate_init_pools(
    baselines_config: str, h0_config: str, h1_config: str, out_dir: Path,
    h3_config: str | None = None, include_h3: bool = False,
    task_indices: tuple[int, ...] = TASK_INDICES_FOR_INIT_POOLS,
    methods: list[str] | None = None,
    filename: str = "motivation_C_init_pools",
) -> tuple[Path, Path]:
    """Not one of the doc's named figures -- a diagnostic requested directly:
    where each initializer's proposed points actually land in the 2D Branin
    input space, against that task's own contour and verified optimum.
    Answers "what does the init pool look like" directly, rather than only
    through the scalar regret it produces (Figure C's own trajectory plot).

    Degenerate-looking pools are not a plotting bug: PriorBestReuse's
    `init_size` points are read back from disk and may coincide (it reuses a
    FIXED set across every held-out task); ContextRegression's `init_size`
    points are the literal same k-NN prediction repeated (see
    core/initializer_baselines.py) and will render as a single marker.

    `methods`/`filename` let a caller render a different subset under a
    different name without touching the default (full-roster) figure that
    motivation_results.md already describes in detail -- e.g. an
    ORPT-H0/H1/H3-focused view, where the three horizon variants get a
    distinct MARKER SHAPE on top of their shared red-family color (see
    style.ABLATION_MARKER), since a scatter has no linestyle to fall back on
    the way the regret-trajectory plots do.
    """
    baselines_cfg = load_config(baselines_config)
    h0_cfg = load_config(h0_config)
    h1_cfg = load_config(h1_config)
    h3_cfg = load_config(h3_config) if h3_config else None
    for cfg, name in ((h0_cfg, "h0"), (h1_cfg, "h1"), *([(h3_cfg, "h3")] if h3_cfg else [])):
        if cfg.manifest.token != baselines_cfg.manifest.token:
            raise ValueError(f"{name} config's task manifest differs from the baselines config")

    if methods is None:
        methods = list(MAIN_METHODS) + (list(SECONDARY_METHODS) if include_h3 else [])

    x1_grid = np.linspace(*BRANIN_BOUNDS[0], 220)
    x2_grid = np.linspace(*BRANIN_BOUNDS[1], 220)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    grid = np.stack([X1, X2], axis=-1)

    n = len(task_indices)
    ncols = min(n, 5)
    nrows = -(-n // ncols)  # ceil division -- wraps into extra rows past 5 panels

    style.apply_rcparams()
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False,
        figsize=(style.FIGSIZE[0] * 2.9 / 3 * ncols, style.FIGSIZE[1] * 1.7 * nrows),
    )
    axes_flat = axes.ravel()
    for ax, task_index in zip(axes_flat, task_indices):
        task = baselines_cfg.heldout_tasks[task_index]
        z = task.oracle(maximize=False).raw(grid)
        z_min = float(z.min())
        levels = np.linspace(z_min, z_min + CONTOUR_WINDOW, 25)
        ax.contourf(X1, X2, z, levels=levels, cmap="Greys", extend="max")
        ax.contour(X1, X2, z, levels=levels[::4], colors=style.MUTED_TEXT, linewidths=0.3, alpha=0.6)

        # Drawn BEFORE (lower zorder than) the method points below: several
        # methods land exactly at/near the optimum -- that's the point being
        # illustrated -- so the star must sit behind them, not occlude them.
        # Sized larger than a method marker so its points still show past the
        # edge of whatever sits on top of its center.
        opt = task.verified
        ax.scatter([opt.x_star[0]], [opt.x_star[1]], marker="*", s=260, color="#FFD700",
                  edgecolor="black", linewidth=0.6, zorder=3, label="Global optimum (verified)")

        for method in methods:
            run_dir, arm, milestone = _resolve(method, baselines_cfg, h0_cfg, h1_cfg, h3_cfg)
            pts = _init_pool(run_dir, arm, milestone, task, baselines_cfg.init_size)
            marker = style.ABLATION_MARKER.get(method, "o")
            ax.scatter(pts[:, 0], pts[:, 1], color=style.arm_color(method), label=LEGEND_LABEL[method],
                      marker=marker, s=32 if marker != "o" else 26,
                      edgecolor="white", linewidth=0.4, zorder=4)

        ax.set_xlim(*BRANIN_BOUNDS[0])
        ax.set_ylim(*BRANIN_BOUNDS[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Task {task_index}")
        ax.set_xlabel("x1")
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("x2")
    for ax in axes_flat[n:]:
        ax.axis("off")  # grid overshoot (n not a multiple of ncols) -- no data for these slots

    last_ax = axes_flat[n - 1]
    # The star is now drawn first (zorder, above) so its legend handle would
    # otherwise come first too -- reorder so it still reads last, same as
    # before that change.
    handles, labels = last_ax.get_legend_handles_labels()
    order = [i for i, l in enumerate(labels) if l != "Global optimum (verified)"]
    order += [i for i, l in enumerate(labels) if l == "Global optimum (verified)"]
    style.place_legend(last_ax, handles=[handles[i] for i in order], labels=[labels[i] for i in order])
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    return style.savefig(fig, out_dir, filename)
