"""fig:scaling (paper/experiments.tex sec:scaling, "Scaling with Completed
Training Tasks"): x-axis = number of completed training tasks (milestone).
Panel (a) = initialization-only performance (summary_incumbent_vs_pool_size
.csv at n_proposals=cfg.init_size), panel (b) = final held-out BO performance
(summary_fixed_target_bo.csv at bo_calls=cfg.oracle_budget, one reference
target_pool_size) -- a relabel/restyle of data
plot_fixed_target_rejection_bo.py's own fig1 already computes, not new
compute logic. One row per domain given (peptide, query_plan), each with its
own (a)/(b) column pair, so up to a 2x2 grid.

Reads incumbent_vs_pool_size.py/fixed_target_rejection_bo.py's summary CSVs
from one or more results dirs -- no new compute. Run those scripts first for
any (arm, milestone) not yet covered. --*-results-dir accepts a
comma-separated list (e.g. every per-milestone GPU shard's own results dir),
same convention as plot_fixed_target_rejection_bo.py -- this is the normal
case, since a full milestone sweep is typically split across GPU shards.

Usage (run from the BOLT repo root):
    python experiments/eval/fig_scaling.py \\
        --peptide-config peptide_experiment/configs/peptide_main_bolt_v2.yaml \\
        --peptide-results-dir experiments/eval/results/main_v2_orpt_vs_bolt__gpu0,experiments/eval/results/main_v2_orpt_vs_bolt__gpu1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paper_labels import DOMAIN_PAPER_NAME, paper_arm
from plot_common import arm_colors, load_concat_csv, plot_arm_line, sorted_arms_from_names, style_axis


def _plot_milestone_panel(ax, df: pd.DataFrame, value_col: str, milestones: list[int], title: str) -> None:
    if df.empty:
        ax.set_title(f"{title} (no data)", color="#0b0b0b")
        style_axis(ax)
        return
    paper_names = sorted_arms_from_names(list({paper_arm(a) for a in df["arm"].unique()}))
    colors = arm_colors(paper_names)
    for internal_arm, arm_df in df.groupby("arm"):
        label = paper_arm(internal_arm)
        stats = arm_df.groupby("milestone")[value_col].agg(["mean", "std"]).reindex(milestones)
        plot_arm_line(ax, milestones, stats["mean"], colors[label], stds=stats["std"], label=label)
    ax.set_xlabel("#completed training tasks (milestone)")
    ax.set_xticks(milestones)
    ax.set_title(title, color="#0b0b0b")
    style_axis(ax)
    ax.legend(frameon=False)


def build_domain_row(cfg, results_dirs: list[Path], task_set: str, target_pool_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    init_only = pd.DataFrame()
    try:
        df = load_concat_csv(results_dirs, "summary_incumbent_vs_pool_size.csv")
    except FileNotFoundError:
        print(f"[fig_scaling] no summary_incumbent_vs_pool_size.csv under {results_dirs} -- "
              "run incumbent_vs_pool_size.py first, init-only panel will be empty")
    else:
        init_only = df[(df["task_set"] == task_set) & (df["n_proposals"] == cfg.init_size)].dropna(subset=["mean_incumbent_mic"])

    final_bo = pd.DataFrame()
    try:
        df = load_concat_csv(results_dirs, "summary_fixed_target_bo.csv")
    except FileNotFoundError:
        print(f"[fig_scaling] no summary_fixed_target_bo.csv under {results_dirs} -- "
              "run fixed_target_rejection_bo.py first, final-BO panel will be empty")
    else:
        final_bo = df[
            (df["task_set"] == task_set) & (df["bo_calls"] == cfg.oracle_budget) & (df["target_pool_size"] == target_pool_size)
        ].dropna(subset=["mean_best_mic"])

    return init_only, final_bo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peptide-config", default=None)
    parser.add_argument("--peptide-results-dir", default=None, help="Comma-separated list of results dirs")
    parser.add_argument("--peptide-task-set", default="heldout100")
    parser.add_argument("--query-plan-config", default=None)
    parser.add_argument("--query-plan-results-dir", default=None, help="Comma-separated list of results dirs")
    parser.add_argument("--query-plan-task-set", default="heldout")
    parser.add_argument("--target-pool-size", type=int, default=None, help="Reference target_pool_size for the final-BO panel (default: each domain's own cfg.init_size)")
    parser.add_argument("--out-dir", default=None, help="Default: experiments/eval/results/fig_scaling")
    args = parser.parse_args()

    from domains import DOMAINS

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "results" / "fig_scaling"

    rows = []
    for domain_name, config_arg, results_dir_arg, task_set in [
        ("peptide", args.peptide_config, args.peptide_results_dir, args.peptide_task_set),
        ("query_plan", args.query_plan_config, args.query_plan_results_dir, args.query_plan_task_set),
    ]:
        if config_arg is None or results_dir_arg is None:
            print(f"[fig_scaling] no --{domain_name.replace('_', '-')}-config/results-dir given, skipping that row")
            continue
        cfg = DOMAINS[domain_name].load_config(config_arg)
        results_dirs = [Path(d.strip()) for d in results_dir_arg.split(",") if d.strip()]
        target = args.target_pool_size if args.target_pool_size is not None else cfg.init_size
        init_only, final_bo = build_domain_row(cfg, results_dirs, task_set, target)
        rows.append((domain_name, cfg, init_only, final_bo))

    if not rows:
        print("[fig_scaling] nothing to plot, exiting")
        return

    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 4.5 * len(rows)), squeeze=False)
    for row_axes, (domain_name, cfg, init_only, final_bo) in zip(axes, rows):
        milestones = list(cfg.milestones)
        _plot_milestone_panel(row_axes[0], init_only, "mean_incumbent_mic", milestones, f"{DOMAIN_PAPER_NAME[domain_name]}: initialization-only")
        _plot_milestone_panel(row_axes[1], final_bo, "mean_best_mic", milestones, f"{DOMAIN_PAPER_NAME[domain_name]}: final held-out BO")
    fig.supylabel("best objective (lower = better)")
    fig.suptitle("Scaling with completed training tasks (fig:scaling)", color="#0b0b0b")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_scaling.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
