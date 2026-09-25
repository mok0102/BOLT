"""fig:main-bo -- best-objective-so-far vs. oracle calls after initialization,
one line per arm, at a single (usually final) milestone.

Unlike the other figures here, this one does not read a summary CSV: it
replays the dense per-task BO trajectories compute/fixed_target_bo.py wrote
and averages them. Deciding which trajectory rows even count requires the
feasibility check that only the domain knows, so this module calls
domains.peptide.running_best_series rather than reducing the CSV itself --
an earlier version took a plain cummin over every row, including infeasible
ones, and was wrong.

Plotted y-values are the domain's own reported objective (MIC for peptide),
matching what compute/incumbent.py records for fig:fewshot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.arms import ArmSpec, bo_work_dir_for
from ..core.spec import DOMAIN_SLUG
from ..domains import peptide
from . import style
from .labels import paper_arm


# arm_specs/main.yaml: STBO/LLAMBO declare "milestones: [0]" only (a
# placeholder, not a real milestone -- see that file's own comment). Querying
# them at the real arms' milestone (e.g. 600) looks for a directory that
# never exists (STBO-600) and silently drops them from the whole figure, so
# they always get their own milestone=0 here regardless of what every other
# arm is plotted at.
MILESTONE_INDEPENDENT_ARMS = ("STBO", "LLAMBO")


def _available_task_ids(cfg, run_dir: Path, task_set: str, arm: str, milestone: int, pool_size: int) -> tuple[Path, set]:
    work_dir = bo_work_dir_for(ArmSpec(arm=arm, milestone=milestone, run_dir=run_dir), task_set, pool_size)
    available = {
        task_id
        for task_id in peptide.task_indices(cfg, task_set)
        if (work_dir / peptide.trajectory_csv_name(task_id)).exists()
    }
    return work_dir, available


def _arm_curve(cfg, work_dir: Path, task_ids, oracle_budget: int) -> pd.Series | None:
    series_list = []
    for task_id in task_ids:
        csv_path = work_dir / peptide.trajectory_csv_name(task_id)
        # Per-task actual pool size, not the blanket pool_size argument: a
        # directory can mix full-target tasks with "fixed budget" tasks
        # whose real init pool is smaller (see main_results_table.py's
        # ROW_SPECS comment) -- using pool_size for those would misindex
        # oracle_calls=0 for exactly those tasks.
        real_pool_size = peptide.read_pool_size(work_dir, task_id)
        series = peptide.running_best_series(cfg, task_id, csv_path, real_pool_size)
        series_list.append(series.loc[:oracle_budget])
    if not series_list:
        return None
    return pd.concat(series_list, axis=1).mean(axis=1, skipna=True)


def _arm_curve_stats(cfg, work_dir: Path, task_ids, oracle_budget: int) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    """Same per-task series as _arm_curve, but also returns a cross-task
    spread at each oracle-calls index (for a std-dev band) -- kept separate
    from _arm_curve rather than changing its return shape, since generate()
    (run_eval.sh's regular pipeline) only ever needs the mean.

    The spread is std(log(MIC)), not std(MIC): every fig1b plot puts MIC on
    a log axis, and a handful of tasks with a much larger best-MIC (still
    well within the same arm) can push a plain linear std well past the
    mean itself, so mean-std goes negative and the band explodes toward
    -inf on a log scale. log-space std, applied multiplicatively around the
    (unchanged) arithmetic mean in _render_fig1b, always stays positive.
    """
    series_list = []
    for task_id in task_ids:
        csv_path = work_dir / peptide.trajectory_csv_name(task_id)
        real_pool_size = peptide.read_pool_size(work_dir, task_id)
        series = peptide.running_best_series(cfg, task_id, csv_path, real_pool_size)
        series_list.append(series.loc[:oracle_budget])
    if not series_list:
        return None, None
    df = pd.concat(series_list, axis=1)
    return df.mean(axis=1, skipna=True), np.log(df).std(axis=1, skipna=True)


def generate(
    milestone: int,
    config_path: Path,
    run_dirs: dict[str, Path],
    task_set: str,
    out_dir: Path,
    target_pool_size: int | None = None,
) -> Path | None:
    cfg = peptide.load_config(str(config_path))
    pool_size = target_pool_size if target_pool_size is not None else cfg.init_size

    # Restrict every arm's mean to the SAME task_idx subset (the intersection
    # across arms present here) instead of each arm's own coverage: coverage
    # varies a lot by arm (self-seeding arms always cover every task; BOLT/
    # ORPT-H1's real rejection-sampled pool means some tasks never reach
    # pool_size feasible candidates, worse at higher milestones), so comparing
    # each arm's own mean compares different task subsets -- see
    # scaling.py::_intersect_and_mean for the same fix on fig:scaling.
    work_dirs: dict[str, tuple[Path, int]] = {}
    available: dict[str, set] = {}
    for arm, run_dir in run_dirs.items():
        arm_milestone = 0 if arm in MILESTONE_INDEPENDENT_ARMS else milestone
        work_dir, ids = _available_task_ids(cfg, run_dir, task_set, arm, arm_milestone, pool_size)
        work_dirs[arm] = (work_dir, arm_milestone)
        if ids:
            available[arm] = ids
        else:
            print(f"[eval2.main_bo] {arm}-{arm_milestone}: no usable task trajectories under {work_dir}, skipping")

    if not available:
        print(f"[eval2.main_bo] milestone={milestone}: nothing to plot, skipping save")
        return None
    shared = set.intersection(*available.values())
    if not shared:
        print(f"[eval2.main_bo] milestone={milestone}: no tasks shared by every arm, skipping")
        return None
    sizes = ", ".join(f"{arm}={len(ids)}" for arm, ids in sorted(available.items()))
    print(f"[eval2.main_bo] milestone={milestone}: intersection={len(shared)} tasks (own coverage: {sizes})")
    shared = sorted(shared)

    style.apply_rcparams()
    fig, ax = plt.subplots(figsize=style.FIGSIZE)

    plotted_any = False
    for arm in run_dirs:
        if arm not in available:
            continue
        work_dir, _arm_milestone = work_dirs[arm]
        curve = _arm_curve(cfg, work_dir, shared, cfg.oracle_budget)
        if curve is None:
            continue
        ax.plot(
            curve.index,
            curve.to_numpy(),
            color=style.arm_color(paper_arm(arm)),
            linewidth=style.LINEWIDTH,
            label=paper_arm(arm),
        )
        plotted_any = True

    if not plotted_any:
        print(f"[eval2.main_bo] milestone={milestone}: nothing to plot, skipping save")
        plt.close(fig)
        return None

    style.style_axis(ax)
    ax.set_xlabel("Oracle calls after initialization")
    ax.set_ylabel("Best objective found")
    style.place_legend(ax)
    return style.savefig(fig, out_dir, f"main_bo_{DOMAIN_SLUG}")


# --- fig:main-bo, paper-Figure-1(b)-style reproduction (ORPT added) ---------
#
# Separate from generate() above (which stays exactly as run_eval.sh's
# regular pipeline calls it, one milestone, one line per arm). This variant
# plots several lines per method -- one per milestone (BOLT/ORPT-MI) or
# n_experts (POGPE/SGPE) -- matching paper/21295's Figure 1(b) peptide panel
# (x=oracle calls, y=MIC, multiple T-indexed lines per method). Reuses the
# runs/lorarank_8 data (oracle_budget=20000, init_size=1000) rather than the
# reduced-budget main run. See /root/.claude/plans/llvm-breezy-jellyfish.md.

FIG1B_ALL_MILESTONES: tuple[int, ...] = (126, 252, 378, 504, 630, 756, 900)
FIG1B_REPRESENTATIVE_MILESTONES: tuple[int, ...] = (378,)  # single selected checkpoint for the decluttered comparison
FIG1B_BOLTORPT_MILESTONES: tuple[int, ...] = (126, 252, 378, 630, 900)  # boltorpt display subset (drops 504/756 for
# readability at 10 lines instead of 14); exact per-milestone numbers for all 7 still live in every render's CSV.
FIG1B_N_EXPERTS: tuple[int, ...] = (5, 10, 20)


def _shade(base_color: str, frac: float, alpha_range: tuple[float, float] = (0.35, 1.0)):
    """base_color (style.ARM_COLOR's hex/alpha) at a given 0..1 position in a
    T-indexed family, lighter for low T and full-strength for high T -- the
    paper's BOLT-T1/BOLT-T2 convention, without needing a distinct color per
    line (every line in a family keeps the arm's one fixed hue)."""
    r, g, b, _a = mcolors.to_rgba(base_color)
    lo, hi = alpha_range
    return (r, g, b, lo + frac * (hi - lo))


FIG1B_FIGSIZE: tuple[float, float] = (style.FIGSIZE[0] * 2, style.FIGSIZE[1] * 2)  # same aspect ratio, 2x pixels at the same DPI
# FIG1B_FIGSIZE is 2x style.FIGSIZE's linear dimensions, so rcParams' plain
# axes.labelsize=9/legend.fontsize=8 (eval2_mok/style.py's convention, shared
# by every other eval2 figure at 1x FIGSIZE) would render text at HALF the
# relative size here -- explicit 2x point sizes keep printed text the same
# apparent size as every other figure in the paper.
FIG1B_AXIS_LABEL_FONTSIZE = 18
FIG1B_LEGEND_FONTSIZE = 16
FIG1B_TICK_LABEL_FONTSIZE = 12  # not the full 2x (16) -- x-axis tick numbers run up to 5 digits
# (20000) and get crowded at 16, so this is a deliberate, smaller-than-2x exception


def _fig1b_series(cfg, bolt_run_dir: Path, orpt_run_dir: Path, task_set: str, pool_size: int,
                   milestones: tuple[int, ...], n_experts: tuple[int, ...], single_milestone: int) -> list[dict]:
    """Collect every line fig1b can plot -- (arm, series_key, work_dir,
    task_ids, color, label) -- computed once and shared by every output
    variant (full/log/boltorpt-zoom), so they never disagree on which tasks
    a given line averages over.

    Does NOT force one global task-id intersection across all ~23 lines --
    that would bottleneck every line down to whatever POGPE/SGPE/STBO's
    deliberately narrow 900-949 targeted range and BOLT/ORPT-MI's highest-
    milestone coverage have in common (measured at 18 tasks), throwing away
    e.g. BOLT-126's 86 real covered tasks for no reason. Instead:
      - BOLT-M / ORPT-MI-M are intersected against EACH OTHER per milestone
        (the only pair actually compared at a shared x=milestone position),
      - every other arm (POGPE/SGPE/MTBO/OptFormer/STBO) plots its own full
        coverage independently, since it has no same-milestone counterpart
        line in this figure to be fairly compared against.
    """
    specs: list[dict] = []

    def _add(arm: str, series_key, work_dir: Path, task_ids, color, label: str) -> None:
        specs.append({"arm": arm, "series_key": series_key, "work_dir": work_dir, "task_ids": task_ids,
                       "color": color, "label": label})

    # BOLT / ORPT-MI: multi-T lines, paired per-milestone intersection.
    # Two accumulator lists (not one interleaved loop) so BOLT's 7 lines
    # group together before ORPT-MI's 7 in plot/legend order, ascending T.
    bolt_specs: list[tuple] = []
    orpt_specs: list[tuple] = []
    n_m = len(milestones)
    for i, m in enumerate(milestones):
        frac = i / (n_m - 1) if n_m > 1 else 1.0
        bolt_dir, bolt_ids = _available_task_ids(cfg, bolt_run_dir, task_set, "BOLT", m, pool_size)
        orpt_dir, orpt_ids = _available_task_ids(cfg, orpt_run_dir, task_set, "ORPT-MI", m, pool_size)
        shared = sorted(bolt_ids & orpt_ids)
        if not shared:
            print(f"[eval2.main_bo.fig1b] BOLT/ORPT-MI milestone={m}: no shared tasks, skipping")
            continue
        print(f"[eval2.main_bo.fig1b] BOLT/ORPT-MI milestone={m}: intersection={len(shared)} "
              f"(BOLT={len(bolt_ids)}, ORPT-MI={len(orpt_ids)})")
        bolt_specs.append(("BOLT", m, bolt_dir, shared, _shade(style.arm_color("BOLT"), frac), f"BOLT-{m}"))
        orpt_specs.append(("ORPT-MI", m, orpt_dir, shared, _shade(style.arm_color("ORPT-MI"), frac), f"ORPT-{m}"))
    for args in bolt_specs + orpt_specs:
        _add(*args)

    # POGPE / SGPE: multi-line by n_experts (already ascending), each its own full coverage.
    n_n = len(n_experts)
    for arm in ("POGPE", "SGPE"):
        for i, n in enumerate(n_experts):
            frac = i / (n_n - 1) if n_n > 1 else 1.0
            work_dir, ids = _available_task_ids(cfg, bolt_run_dir, task_set, arm, n, pool_size)
            if not ids:
                print(f"[eval2.main_bo.fig1b] {arm}-{n}: no usable trajectories, skipping")
                continue
            _add(arm, n, work_dir, sorted(ids), _shade(style.arm_color(arm), frac), f"{paper_arm(arm)}-{n}")

    # MTBO / OptFormer: single representative-milestone line, own coverage.
    for arm in ("MTBO", "OptFormer"):
        work_dir, ids = _available_task_ids(cfg, bolt_run_dir, task_set, arm, single_milestone, pool_size)
        if not ids:
            print(f"[eval2.main_bo.fig1b] {arm}-{single_milestone}: no usable trajectories, skipping")
            continue
        _add(arm, single_milestone, work_dir, sorted(ids), style.arm_color(arm), paper_arm(arm))

    # STBO: self-seeding, no milestone axis (arm_specs/fig1b.yaml: milestones=[0]).
    work_dir, ids = _available_task_ids(cfg, bolt_run_dir, task_set, "STBO", 0, pool_size)
    if ids:
        _add("STBO", 0, work_dir, sorted(ids), style.arm_color("STBO"), paper_arm("STBO"))
    else:
        print("[eval2.main_bo.fig1b] STBO-0: no usable trajectories, skipping")

    return specs


# Milestone (T) is an ordered quantity, so color encodes that order via a
# perceptually-uniform, colorblind-safe sequential colormap (viridis -- the
# standard modern choice for print/grayscale-safe publication figures)
# rather than a small set of hand-picked hues. Shared between BOLT and
# ORPT-MI at the same milestone, so "same color" means "same T" at a
# glance. Linestyle (not hue) carries the BOLT-vs-ORPT distinction here: an
# intentional one-off exception to style.py's "color = arm identity"
# convention, since this figure's whole point is comparing the SAME T
# across two arms, not telling ~20 different arms apart.
FIG1B_BOLTORPT_LINESTYLE: dict[str, str] = {"BOLT": "--", "ORPT-MI": "-"}


def _boltorpt_colors(milestones: tuple[int, ...]) -> dict[int, str]:
    n = len(milestones)
    cmap = plt.get_cmap("viridis")
    return {m: mcolors.to_hex(cmap(i / (n - 1) if n > 1 else 1.0)) for i, m in enumerate(milestones)}


def _boltorpt_specs(cfg, bolt_run_dir: Path, orpt_run_dir: Path, task_set: str, pool_size: int,
                     milestones: tuple[int, ...] = FIG1B_ALL_MILESTONES) -> list[dict]:
    """BOLT/ORPT-MI at every milestone -- this figure is the dedicated,
    full-detail BOLT-vs-ORPT-MI comparison (the decluttered main fig1b
    figure only keeps FIG1B_REPRESENTATIVE_MILESTONES) -- colored by
    milestone, not by arm, so paired lines read as directly comparable at a
    glance."""
    colors = _boltorpt_colors(milestones)
    specs: list[dict] = []
    for m in milestones:
        bolt_dir, bolt_ids = _available_task_ids(cfg, bolt_run_dir, task_set, "BOLT", m, pool_size)
        orpt_dir, orpt_ids = _available_task_ids(cfg, orpt_run_dir, task_set, "ORPT-MI", m, pool_size)
        shared = sorted(bolt_ids & orpt_ids)
        if not shared:
            print(f"[eval2.main_bo.fig1b.boltorpt] milestone={m}: no shared tasks, skipping")
            continue
        print(f"[eval2.main_bo.fig1b.boltorpt] milestone={m}: intersection={len(shared)} "
              f"(BOLT={len(bolt_ids)}, ORPT-MI={len(orpt_ids)})")
        color = colors[m]
        specs.append({"arm": "BOLT", "series_key": m, "work_dir": bolt_dir, "task_ids": shared,
                       "color": color, "linestyle": FIG1B_BOLTORPT_LINESTYLE["BOLT"], "label": f"BOLT-{m}"})
        specs.append({"arm": "ORPT-MI", "series_key": m, "work_dir": orpt_dir, "task_ids": shared,
                       "color": color, "linestyle": FIG1B_BOLTORPT_LINESTYLE["ORPT-MI"], "label": f"ORPT-{m}"})
    return specs


def _render_fig1b(cfg, specs: list[dict], out_dir: Path, name: str,
                   arms: tuple[str, ...] | None = None, yscale: str = "log", xscale: str = "linear",
                   show_std: bool = False) -> Path | None:
    style.apply_rcparams()
    fig, ax = plt.subplots(figsize=FIG1B_FIGSIZE)
    rows: list[dict] = []
    plotted_any = False

    for spec in specs:
        if arms is not None and spec["arm"] not in arms:
            continue
        mean, log_std = _arm_curve_stats(cfg, spec["work_dir"], spec["task_ids"], cfg.oracle_budget)
        if mean is None:
            continue
        if show_std:
            # Multiplicative band (mean * exp(-+log_std)) around the
            # unchanged arithmetic mean -- see _arm_curve_stats for why this
            # is log_std, not std. alpha=0.15: visible without drowning out
            # the mean lines underneath, since several arms' bands overlap.
            spread = np.exp(log_std.to_numpy())
            mean_arr = mean.to_numpy()
            ax.fill_between(mean.index, mean_arr / spread, mean_arr * spread,
                             color=spec["color"], alpha=0.15, linewidth=0)
        ax.plot(mean.index, mean.to_numpy(), color=spec["color"], linestyle=spec.get("linestyle", "-"),
                linewidth=style.LINEWIDTH, label=spec["label"])
        plotted_any = True
        for oracle_calls, best_mic, log_std_mic in zip(mean.index, mean.to_numpy(), log_std.to_numpy()):
            rows.append({
                "arm": spec["arm"], "milestone": spec["series_key"], "n_tasks": len(spec["task_ids"]),
                "oracle_calls": oracle_calls, "mean_best_mic": best_mic, "log_std_best_mic": log_std_mic,
            })

    if not plotted_any:
        print(f"[eval2.main_bo.fig1b] {name}: nothing to plot, skipping save")
        plt.close(fig)
        return None

    style.style_axis(ax)
    ax.set_xlabel("Oracle calls after initialization", fontsize=FIG1B_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Best MIC found", fontsize=FIG1B_AXIS_LABEL_FONTSIZE)
    # which="both": the log y-axis has minor ticks too (2,3,4,6x10^1, ...)
    # -- tick_params defaults to which="major" only, which left minor tick
    # labels at rcParams' unscaled 8pt while major ticks (10^1, 10^2) got
    # FIG1B_TICK_LABEL_FONTSIZE, a visible size mismatch between them.
    ax.tick_params(axis="both", which="both", labelsize=FIG1B_TICK_LABEL_FONTSIZE)
    if yscale != "linear":
        ax.set_yscale(yscale)
    if xscale != "linear":
        # plain "log" can't represent oracle_calls=0 (the init-pool point,
        # a real and informative part of every curve) -- symlog keeps a
        # small linear region around 0 and logs everything past it instead
        # of silently dropping that first point. symlog mirrors negative
        # values by default; oracle_calls is never negative, so clip that
        # unused mirror half out of view instead of wasting half the plot.
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(left=0)
    style.place_legend(ax, fontsize=FIG1B_LEGEND_FONTSIZE)
    path = style.savefig(fig, out_dir, name)
    style.save_csv(pd.DataFrame(rows), out_dir, name)
    return path


def generate_fig1b(
    config_path: Path,
    bolt_run_dir: Path,
    orpt_run_dir: Path,
    task_set: str,
    out_dir: Path,
    target_pool_size: int | None = None,
    representative_milestones: tuple[int, ...] = FIG1B_REPRESENTATIVE_MILESTONES,
    all_milestones: tuple[int, ...] = FIG1B_BOLTORPT_MILESTONES,
    n_experts: tuple[int, ...] = FIG1B_N_EXPERTS,
    single_milestone: int = 900,
) -> dict[str, Path | None]:
    """Writes three PNG+CSV pairs sharing the same underlying series (see
    _fig1b_series). Log y-axis is the default (baselines sit at 26-44 MIC,
    BOLT/ORPT-MI at 6-16 -- a linear axis compresses BOLT/ORPT-MI's own
    T-driven improvement almost flat):
      - main_bo_fig1b_<domain>: every arm (BOLT/ORPT-MI only at
        representative_milestones -- the full per-T trend belongs on the
        dedicated boltorpt figure below, not repeated here among ~20 other
        lines), log y-axis.
      - main_bo_fig1b_<domain>_linear: same lines, linear y-axis (kept for
        reference alongside the log default).
      - main_bo_fig1b_<domain>_boltorpt: BOLT/ORPT-MI only, at
        all_milestones (a display-density-reduced subset, FIG1B_BOLTORPT_
        MILESTONES by default -- not literally every trained checkpoint;
        see that constant), log y-axis auto-scaled to just their range,
        colored by milestone (not arm) so same-T lines are directly
        comparable -- see _boltorpt_specs.
    """
    cfg = peptide.load_config(str(config_path))
    pool_size = target_pool_size if target_pool_size is not None else cfg.init_size
    specs = _fig1b_series(cfg, bolt_run_dir, orpt_run_dir, task_set, pool_size, representative_milestones, n_experts, single_milestone)
    boltorpt_specs = _boltorpt_specs(cfg, bolt_run_dir, orpt_run_dir, task_set, pool_size, milestones=all_milestones)

    base = f"main_bo_fig1b_{DOMAIN_SLUG}"
    return {
        "full": _render_fig1b(cfg, specs, out_dir, base),
        "full_linear": _render_fig1b(cfg, specs, out_dir, f"{base}_linear", yscale="linear"),
        "boltorpt": _render_fig1b(cfg, boltorpt_specs, out_dir, f"{base}_boltorpt"),
    }
