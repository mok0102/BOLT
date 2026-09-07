"""tab:main-bo-summary (paper/experiments.tex sec:main-results, optional
summary table -- "Include only if it adds useful information beyond the
trajectory figure... Prefer a compact table."): Domain | Method |
Initialization | Final BO, rows = {query_plan, peptide} x {BOLT, ORPT}, one
reference milestone (default: max(cfg.milestones)). New label -- the tex's
own commented-out table sketch doesn't define one.

Reads incumbent_vs_pool_size.py/fixed_target_rejection_bo.py's summary CSVs
from one or more results dirs (comma-separated, same convention as
plot_fixed_target_rejection_bo.py) -- no new compute.

Either domain's config/results-dir may be omitted -- its two rows are
rendered as "--" with a footnote, not silently dropped (so the table's shape
stays fixed regardless of which domains have data yet).

Usage (run from the BOLT repo root):
    python experiments/eval/tab_main_bo_summary.py \\
        --peptide-config peptide_experiment/configs/peptide_main_bolt_v2.yaml \\
        --peptide-results-dir experiments/eval/results/main_v2_orpt_vs_bolt__gpu0,...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper_labels import DOMAIN_PAPER_NAME, paper_arm
from plot_common import load_concat_csv

TABLE_ARMS = ("BOLT", "ORPT")  # the paper's own table sketch: just the core comparison, not every baseline


def _lookup_one(results_dirs: list[Path], filename: str) -> pd.DataFrame:
    try:
        return load_concat_csv(results_dirs, filename)
    except FileNotFoundError:
        print(f"[tab_main_bo_summary] no {filename} under {results_dirs}")
        return pd.DataFrame()


def domain_rows(domain_name: str, cfg, results_dirs: list[Path], task_set: str, milestone: int) -> list[dict]:
    init_df = _lookup_one(results_dirs, "summary_incumbent_vs_pool_size.csv")
    final_df = _lookup_one(results_dirs, "summary_fixed_target_bo.csv")

    rows = []
    for method in TABLE_ARMS:
        init_val = None
        if not init_df.empty:
            sub = init_df[
                (init_df["task_set"] == task_set) & (init_df["milestone"] == milestone)
                & (init_df["n_proposals"] == cfg.init_size) & (init_df["arm"].apply(paper_arm) == method)
            ]
            if not sub.empty:
                init_val = float(sub["mean_incumbent_mic"].iloc[0])

        final_val = None
        if not final_df.empty:
            sub = final_df[
                (final_df["task_set"] == task_set) & (final_df["milestone"] == milestone)
                & (final_df["bo_calls"] == cfg.oracle_budget) & (final_df["target_pool_size"] == cfg.init_size)
                & (final_df["arm"].apply(paper_arm) == method)
            ]
            if not sub.empty:
                final_val = float(sub["mean_best_mic"].iloc[0])

        rows.append({
            "domain": DOMAIN_PAPER_NAME[domain_name], "method": method,
            "initialization": init_val, "final_bo": final_val,
        })
    return rows


def to_latex(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Domain & Method & Initialization $\downarrow$ & Final BO $\downarrow$ \\",
        r"\midrule",
    ]
    prev_domain = None
    for row in rows:
        domain_cell = row["domain"] if row["domain"] != prev_domain else ""
        prev_domain = row["domain"]
        init_cell = f"{row['initialization']:.2f}" if row["initialization"] is not None else "--"
        final_cell = f"{row['final_bo']:.2f}" if row["final_bo"] is not None else "--"
        lines.append(f"{domain_cell} & {row['method']} & {init_cell} & {final_cell} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peptide-config", default=None)
    parser.add_argument("--peptide-results-dir", default=None)
    parser.add_argument("--peptide-task-set", default="heldout100")
    parser.add_argument("--query-plan-config", default=None)
    parser.add_argument("--query-plan-results-dir", default=None)
    parser.add_argument("--query-plan-task-set", default="heldout")
    parser.add_argument("--milestone", type=int, default=None, help="Reference milestone (default: each domain's own max(cfg.milestones))")
    parser.add_argument("--out-dir", default=None, help="Default: experiments/eval/results/tab_main_bo_summary")
    args = parser.parse_args()

    from domains import DOMAINS

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "results" / "tab_main_bo_summary"

    all_rows = []
    for domain_name, config_arg, results_dir_arg, task_set in [
        ("query_plan", args.query_plan_config, args.query_plan_results_dir, args.query_plan_task_set),
        ("peptide", args.peptide_config, args.peptide_results_dir, args.peptide_task_set),
    ]:
        if config_arg is None or results_dir_arg is None:
            print(f"[tab_main_bo_summary] no --{domain_name.replace('_', '-')}-config/results-dir given, "
                  f"{DOMAIN_PAPER_NAME[domain_name]} rows will be all '--'")
            all_rows += [{"domain": DOMAIN_PAPER_NAME[domain_name], "method": m, "initialization": None, "final_bo": None} for m in TABLE_ARMS]
            continue
        cfg = DOMAINS[domain_name].load_config(config_arg)
        results_dirs = [Path(d.strip()) for d in results_dir_arg.split(",") if d.strip()]
        milestone = args.milestone if args.milestone is not None else max(cfg.milestones)
        all_rows += domain_rows(domain_name, cfg, results_dirs, task_set, milestone)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_dir / "tab_main_bo_summary.csv", index=False)
    (out_dir / "tab_main_bo_summary.tex").write_text(to_latex(all_rows) + "\n")
    print(f"Wrote {out_dir}/tab_main_bo_summary.csv and .tex")


if __name__ == "__main__":
    main()
