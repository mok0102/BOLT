"""fig:main-bo (paper/experiments.tex sec:main-results, "Full-budget
optimization"): two panels (query-plan, peptide), x-axis = test-task oracle
calls, y-axis = best-so-far objective, one curve per arm (ORPT, BOLT, STBO,
MTBO, POGPE/SGPE, OptFormer, LLAMBO -- whichever are present in the given
manifest at the reference milestone). oracle_calls=0 marks the paper's own
"end of initialization / optimization start".

Reads the full per-task LOLBO trajectory CSVs fixed_target_rejection_bo.py
already leaves on disk under
<run_dir>/eval_fixed_target_bo/<task_set>/<arm>-<milestone>__target<T>/... --
no new compute, just a dense re-aggregation via domain.running_best_series.
Run fixed_target_rejection_bo.py first for any (arm, milestone, target) not
yet covered.

Either domain's config/manifest may be omitted -- that panel is skipped (with
a printed note), so this can produce just the peptide panel today and the
query-plan panel once that domain has real ORPT checkpoints.

Usage (run from the BOLT repo root):
    python experiments/eval/fig_main_bo.py \\
        --peptide-config peptide_experiment/configs/peptide_main_bolt_v2.yaml \\
        --peptide-manifest experiments/eval/manifests/main_v2_orpt_vs_bolt.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import ModelSpec, load_manifest
from domains import DOMAINS, Domain
from paper_labels import DOMAIN_PAPER_NAME, paper_arm
from plot_common import arm_colors, sorted_arms_from_names, style_axis

FIG_LABEL = "fig:main-bo"


def _specs_for_milestone(specs: list[ModelSpec], milestone: int) -> list[ModelSpec]:
    """specs at the reference milestone, plus any arm that only ever appears
    at a single milestone across the whole manifest (STBO/LLAMBO-style
    self-seeding baselines, whose milestone field is an unused placeholder --
    same "milestone-independent" concept plot_common.py's is_milestone_independent
    applies to per-task summary CSVs, applied here directly to the manifest)."""
    milestones_by_arm: dict[str, set[int]] = {}
    for s in specs:
        milestones_by_arm.setdefault(s.arm, set()).add(s.milestone)
    return [s for s in specs if len(milestones_by_arm[s.arm]) == 1 or s.milestone == milestone]


def _mean_series_for_spec(
    domain: Domain, cfg, spec: ModelSpec, task_set: str, target: int, max_oracle_calls: int,
) -> pd.Series | None:
    work_dir = spec.run_dir / "eval_fixed_target_bo" / task_set / f"{spec.arm}-{spec.milestone}__target{target}"
    task_ids = domain.task_indices(cfg, task_set)
    series_list = []
    for task_id in task_ids:
        csv_path = work_dir / domain.trajectory_csv_name(task_id)
        if not csv_path.exists():
            continue
        series = domain.running_best_series(cfg, task_id, csv_path, target)
        # Real trajectory CSVs have been observed with far more rows than
        # cfg.oracle_budget would suggest (one task's collected-data CSV can
        # run 2-3x longer than another's, same arm/milestone) -- capping here
        # keeps the mean-across-tasks curve from being dominated, past the
        # nominal budget, by whichever one or two tasks happen to have an
        # unusually long trajectory on disk.
        series_list.append(series.loc[:max_oracle_calls])
    if not series_list:
        print(f"[fig_main_bo] {spec.arm}-{spec.milestone}: no trajectory CSVs found under {work_dir}, skipping")
        return None
    return pd.concat(series_list, axis=1).mean(axis=1, skipna=True)


def build_panel(
    domain: Domain, cfg, specs: list[ModelSpec], task_set: str, milestone: int, target: int,
) -> dict[str, pd.Series]:
    selected = _specs_for_milestone(specs, milestone)
    curves: dict[str, pd.Series] = {}
    for spec in selected:
        series = _mean_series_for_spec(domain, cfg, spec, task_set, target, cfg.oracle_budget)
        if series is not None:
            curves[spec.arm] = series
    return curves


def plot_panel(ax, curves: dict[str, pd.Series], domain_label: str) -> None:
    # Color/order/legend are keyed on the paper-facing name (paper_arm()),
    # not the internal arm identifier -- so ORPT-MI/ORPT-H1 (same method,
    # different points in time) always render in ORPT's established red.
    paper_names = sorted_arms_from_names(list({paper_arm(a) for a in curves}))
    colors = arm_colors(paper_names)
    for arm, series in curves.items():
        label = paper_arm(arm)
        ax.plot(series.index, series.values, color=colors[label], linewidth=2, label=label)
    ax.axvline(0, color="#c3c2b7", linewidth=1, linestyle=":")
    ax.set_xlabel("test-task oracle calls")
    ax.set_title(domain_label, color="#0b0b0b")
    style_axis(ax)
    ax.legend(frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peptide-config", default=None)
    parser.add_argument("--peptide-manifest", default=None)
    parser.add_argument("--peptide-task-set", default="heldout100")
    parser.add_argument("--query-plan-config", default=None)
    parser.add_argument("--query-plan-manifest", default=None)
    parser.add_argument("--query-plan-task-set", default="heldout")
    parser.add_argument("--milestone", type=int, default=None, help="Reference milestone (default: each domain's own max(cfg.milestones))")
    parser.add_argument("--target-pool-size", type=int, default=None, help="Reference target_pool_size (default: each domain's own cfg.init_size)")
    parser.add_argument("--out-dir", default=None, help="Default: experiments/eval/results/fig_main_bo")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "results" / "fig_main_bo"

    panels = []
    for domain_name, config_arg, manifest_arg, task_set in [
        ("query_plan", args.query_plan_config, args.query_plan_manifest, args.query_plan_task_set),
        ("peptide", args.peptide_config, args.peptide_manifest, args.peptide_task_set),
    ]:
        if config_arg is None or manifest_arg is None:
            print(f"[fig_main_bo] no --{domain_name.replace('_', '-')}-config/manifest given, skipping that panel")
            continue
        domain = DOMAINS[domain_name]
        cfg = domain.load_config(config_arg)
        specs = load_manifest(manifest_arg)
        milestone = args.milestone if args.milestone is not None else max(cfg.milestones)
        target = args.target_pool_size if args.target_pool_size is not None else cfg.init_size
        curves = build_panel(domain, cfg, specs, task_set, milestone, target)
        if curves:
            panels.append((domain_name, curves))
        else:
            print(f"[fig_main_bo] no data at all for domain={domain_name}, milestone={milestone}, target={target}")

    if not panels:
        print("[fig_main_bo] nothing to plot, exiting")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.5), squeeze=False)
    for ax, (domain_name, curves) in zip(axes[0], panels):
        plot_panel(ax, curves, DOMAIN_PAPER_NAME[domain_name])
    fig.supylabel("best-so-far objective (lower = better)")
    fig.suptitle(f"Full-budget optimization ({FIG_LABEL})", color="#0b0b0b")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_main_bo.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
