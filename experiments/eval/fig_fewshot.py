"""fig:fewshot (paper/experiments.tex sec:main-results, "Initialization and
few-shot proposal quality"): two panels, one per domain, x-axis = number of
proposal samples k, y-axis = best-of-k objective, one curve per arm at a
single reference milestone (default: max(cfg.milestones), i.e. the fully
trained model). Optional horizontal "final STBO"/"final MTBO" reference
lines (peptide only -- those baselines don't exist for query_plan yet).

Reads incumbent_vs_pool_size.py's summary_incumbent_vs_pool_size.csv (and,
for the optional reference lines, fixed_target_rejection_bo.py's
summary_fixed_target_bo.csv) from one or more results dirs -- no new compute.
Run those scripts first for any (arm, milestone) not yet covered.
--*-results-dir accepts a comma-separated list (e.g. every per-milestone GPU
shard's own results dir), same convention as plot_fixed_target_rejection_bo.py.

Either domain's config/results-dir may be omitted -- that panel is skipped
(with a printed note).

Usage (run from the BOLT repo root):
    python experiments/eval/fig_fewshot.py \\
        --peptide-config peptide_experiment/configs/peptide_main_bolt_v2.yaml \\
        --peptide-results-dir experiments/eval/results/main_v2_orpt_vs_bolt__gpu0,experiments/eval/results/main_v2_orpt_vs_bolt__gpu1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paper_labels import DOMAIN_PAPER_NAME, paper_arm
from plot_common import arm_colors, load_concat_csv, sorted_arms_from_names, style_axis

REFERENCE_ARMS = ("STBO", "MTBO")  # peptide-only, drawn as horizontal reference lines when present


def build_panel(domain_name: str, cfg, results_dirs: list[Path], task_set: str, milestone: int) -> tuple[dict, dict]:
    try:
        df = load_concat_csv(results_dirs, "summary_incumbent_vs_pool_size.csv")
    except FileNotFoundError:
        print(f"[fig_fewshot] no summary_incumbent_vs_pool_size.csv under {results_dirs} -- "
              f"run incumbent_vs_pool_size.py first, skipping {domain_name}")
        return {}, {}
    sub = df[(df["task_set"] == task_set) & (df["milestone"] == milestone)].dropna(subset=["mean_incumbent_mic"])

    curves: dict[str, pd.Series] = {}
    for arm, arm_df in sub.groupby("arm"):
        curves[arm] = arm_df.set_index("n_proposals")["mean_incumbent_mic"].sort_index()

    references: dict[str, float] = {}
    if domain_name == "peptide":
        try:
            ft = load_concat_csv(results_dirs, "summary_fixed_target_bo.csv")
        except FileNotFoundError:
            print(f"[fig_fewshot] no summary_fixed_target_bo.csv under {results_dirs} -- final STBO/MTBO reference lines omitted")
        else:
            final = ft[(ft["task_set"] == task_set) & (ft["bo_calls"] == cfg.oracle_budget)]
            for arm in REFERENCE_ARMS:
                arm_rows = final[final["arm"] == arm]
                if not arm_rows.empty:
                    references[arm] = float(arm_rows["mean_best_mic"].iloc[0])
    return curves, references


def plot_panel(ax, curves: dict[str, pd.Series], references: dict[str, float], domain_label: str) -> None:
    if not curves and not references:
        ax.set_title(f"{domain_label} (no data)", color="#0b0b0b")
        style_axis(ax)
        return
    paper_names = sorted_arms_from_names(list({paper_arm(a) for a in curves} | set(references)))
    colors = arm_colors(paper_names)
    for arm, series in curves.items():
        label = paper_arm(arm)
        ax.plot(series.index, series.values, color=colors[label], linewidth=2, marker="o", markersize=6, label=label)
    for arm, value in references.items():
        ax.axhline(value, color=colors[paper_arm(arm)], linewidth=2, linestyle="--", label=f"final {paper_arm(arm)}")
    ax.set_xscale("log")
    ax.set_xlabel("number of proposal samples k")
    ax.set_title(domain_label, color="#0b0b0b")
    style_axis(ax)
    ax.legend(frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peptide-config", default=None)
    parser.add_argument("--peptide-results-dir", default=None, help="Comma-separated list of results dirs")
    parser.add_argument("--peptide-task-set", default="heldout100")
    parser.add_argument("--query-plan-config", default=None)
    parser.add_argument("--query-plan-results-dir", default=None, help="Comma-separated list of results dirs")
    parser.add_argument("--query-plan-task-set", default="heldout")
    parser.add_argument("--milestone", type=int, default=None, help="Reference milestone (default: each domain's own max(cfg.milestones))")
    parser.add_argument("--out-dir", default=None, help="Default: experiments/eval/results/fig_fewshot")
    args = parser.parse_args()

    from domains import DOMAINS

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "results" / "fig_fewshot"

    panels = []
    for domain_name, config_arg, results_dir_arg, task_set in [
        ("query_plan", args.query_plan_config, args.query_plan_results_dir, args.query_plan_task_set),
        ("peptide", args.peptide_config, args.peptide_results_dir, args.peptide_task_set),
    ]:
        if config_arg is None or results_dir_arg is None:
            print(f"[fig_fewshot] no --{domain_name.replace('_', '-')}-config/results-dir given, skipping that panel")
            continue
        cfg = DOMAINS[domain_name].load_config(config_arg)
        results_dirs = [Path(d.strip()) for d in results_dir_arg.split(",") if d.strip()]
        milestone = args.milestone if args.milestone is not None else max(cfg.milestones)
        curves, references = build_panel(domain_name, cfg, results_dirs, task_set, milestone)
        panels.append((domain_name, curves, references))

    if not any(curves or references for _, curves, references in panels):
        print("[fig_fewshot] nothing to plot, exiting")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.5), squeeze=False)
    for ax, (domain_name, curves, references) in zip(axes[0], panels):
        plot_panel(ax, curves, references, DOMAIN_PAPER_NAME[domain_name])
    fig.supylabel("best-of-k objective (lower = better)")
    fig.suptitle("Initialization and few-shot proposal quality (fig:fewshot)", color="#0b0b0b")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_fewshot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
