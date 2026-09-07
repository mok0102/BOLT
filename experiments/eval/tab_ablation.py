"""tab:ablation (paper/experiments.tex sec:ablations, "Ablation Study"):
Training signal | Initialization | Final BO, three rows in the paper's own
order (Supervised proposal -> Zero-step outcome ranking -> One-step outcome
ranking), sourced from experiments/eval/manifests/ablation_h0_vs_h1.yaml's
results (arms BOLT/ORPT-H0/ORPT-H1, mapped via
paper_labels.ABLATION_ARM_TO_SIGNAL). New label -- the tex's own
commented-out table sketch doesn't define one.

Full main-experiment scale (see paper_labels.ABLATION_SCALE_NOTE): BOLT and
ORPT-H1 rows are the exact same checkpoints as the main results
(runs/peptide_main_bolt_v2/runs/peptide_main_orpt_h1_v2) -- if the main
experiment's own eval pipeline has already run, this script's BOLT/ORPT-H1
numbers come for free (idempotent reuse). PREREQUISITE (as of this writing,
verify before trusting the output): ORPT-H0 (the zero-step ablation arm) has
never been trained at any scale -- no checkpoint exists anywhere on disk.
Train it via experiments/eval/run_main_bolt_vs_orpt_train.sh
(peptide_experiment/configs/peptide_ablation_orpt_h0.yaml, mi_bo_steps=0),
then run, against manifests/ablation_h0_vs_h1.yaml:
    python experiments/eval/generate_raw_proposals.py --domain peptide \\
        --config peptide_experiment/configs/peptide_ablation_orpt_h0.yaml \\
        --manifest experiments/eval/manifests/ablation_h0_vs_h1.yaml --task-sets heldout100
    python experiments/eval/incumbent_vs_pool_size.py --domain peptide \\
        --config peptide_experiment/configs/peptide_ablation_orpt_h0.yaml \\
        --manifest experiments/eval/manifests/ablation_h0_vs_h1.yaml --task-sets heldout100
    python experiments/eval/fixed_target_rejection_bo.py --domain peptide \\
        --config peptide_experiment/configs/peptide_ablation_orpt_h0.yaml \\
        --manifest experiments/eval/manifests/ablation_h0_vs_h1.yaml --task-sets heldout100
before this script produces real ORPT-H0 numbers -- this is real training +
compute work, not something this script can shortcut. (Any one of the three
configs in the manifest works as the shared --config here -- they all agree
on milestones/init_size/oracle_budget by construction.)

Usage (run from the BOLT repo root):
    python experiments/eval/tab_ablation.py \\
        --config peptide_experiment/configs/peptide_main_bolt_v2.yaml \\
        --results-dir experiments/eval/results/ablation_h0_vs_h1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper_labels import ABLATION_ARM_TO_SIGNAL, ABLATION_ROW_ORDER, ABLATION_SCALE_NOTE
from plot_common import load_concat_csv


def build_rows(cfg, results_dirs: list[Path], task_set: str, milestone: int) -> list[dict]:
    try:
        init_df = load_concat_csv(results_dirs, "summary_incumbent_vs_pool_size.csv")
    except FileNotFoundError:
        print(f"[tab_ablation] no summary_incumbent_vs_pool_size.csv under {results_dirs}")
        init_df = pd.DataFrame()
    try:
        final_df = load_concat_csv(results_dirs, "summary_fixed_target_bo.csv")
    except FileNotFoundError:
        print(f"[tab_ablation] no summary_fixed_target_bo.csv under {results_dirs}")
        final_df = pd.DataFrame()

    by_signal: dict[str, dict] = {signal: {"training_signal": signal, "initialization": None, "final_bo": None} for signal in ABLATION_ROW_ORDER}
    for internal_arm, signal in ABLATION_ARM_TO_SIGNAL.items():
        if not init_df.empty:
            sub = init_df[
                (init_df["task_set"] == task_set) & (init_df["milestone"] == milestone)
                & (init_df["n_proposals"] == cfg.init_size) & (init_df["arm"] == internal_arm)
            ]
            if not sub.empty:
                by_signal[signal]["initialization"] = float(sub["mean_incumbent_mic"].iloc[0])
        if not final_df.empty:
            sub = final_df[
                (final_df["task_set"] == task_set) & (final_df["milestone"] == milestone)
                & (final_df["bo_calls"] == cfg.oracle_budget) & (final_df["target_pool_size"] == cfg.init_size)
                & (final_df["arm"] == internal_arm)
            ]
            if not sub.empty:
                by_signal[signal]["final_bo"] = float(sub["mean_best_mic"].iloc[0])

    return [by_signal[signal] for signal in ABLATION_ROW_ORDER]


def to_latex(rows: list[dict]) -> str:
    lines = [
        f"% {ABLATION_SCALE_NOTE}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Training signal & Initialization $\downarrow$ & Final BO $\downarrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        init_cell = f"{row['initialization']:.2f}" if row["initialization"] is not None else "--"
        final_cell = f"{row['final_bo']:.2f}" if row["final_bo"] is not None else "--"
        lines.append(f"{row['training_signal']} & {init_cell} & {final_cell} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", required=True, help="Comma-separated list of results dirs")
    parser.add_argument("--task-set", default="heldout100")
    parser.add_argument("--milestone", type=int, default=None, help="Reference milestone (default: max(cfg.milestones))")
    parser.add_argument("--out-dir", default=None, help="Default: experiments/eval/results/tab_ablation")
    args = parser.parse_args()

    from domains import DOMAINS

    cfg = DOMAINS["peptide"].load_config(args.config)
    results_dirs = [Path(d.strip()) for d in args.results_dir.split(",") if d.strip()]
    milestone = args.milestone if args.milestone is not None else max(cfg.milestones)

    rows = build_rows(cfg, results_dirs, args.task_set, milestone)

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "results" / "tab_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "tab_ablation.csv", index=False)
    (out_dir / "tab_ablation.tex").write_text(to_latex(rows) + "\n")
    print(f"Wrote {out_dir}/tab_ablation.csv and .tex")
    print(f"NOTE: {ABLATION_SCALE_NOTE}")


if __name__ == "__main__":
    main()
