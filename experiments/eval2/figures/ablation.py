"""tab:ablation -- one bare LaTeX tabular (+ .csv twin): rows =
{Supervised proposal (BOLT), Zero-step outcome ranking (ORPT-H0), One-step
outcome ranking (ORPT-H1)}, columns = {Initialization, Final BO}.

Column semantics: "Initialization" = mean_incumbent_mic at n_proposals ==
target_pool_size (the actual init pool size BO starts from, not an arbitrary
smaller k); "Final BO" = mean_best_mic at bo_calls == the oracle budget. Both
slice values default to what the summary CSVs actually contain rather than a
hardcoded copy. Any scale caveat (e.g. "H0/H1 use independent trajectory
chains") is printed to stdout, never written into the .tex output -- that
belongs in the paper's prose, not the artifact.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.results_io import load_summary_fixed_target_bo, load_summary_incumbent
from ..core.spec import DOMAIN_SLUG
from .labels import ABLATION_ARM_TO_ROW, ABLATION_ROW_ORDER

ABLATION_ARMS = ("BOLT", "ORPT-H0", "ORPT-H1")


def _pick(df, column: str, requested: int | None) -> int | None:
    if requested is not None:
        return requested
    values = sorted(df[column].dropna().unique())
    return int(values[-1]) if values else None


def build_rows(
    results_dirs: str | Path | list[str | Path],
    task_set: str,
    milestone: int,
    target_pool_size: int | None = None,
    oracle_budget: int | None = None,
) -> list[dict]:
    init_df = load_summary_incumbent(results_dirs)
    final_df = load_summary_fixed_target_bo(results_dirs)
    target = _pick(final_df, "target_pool_size", target_pool_size) if not final_df.empty else target_pool_size
    budget = _pick(final_df, "bo_calls", oracle_budget) if not final_df.empty else oracle_budget

    by_row = {row: {"training_signal": row, "initialization": None, "final_bo": None} for row in ABLATION_ROW_ORDER}
    for arm in ABLATION_ARMS:
        row = ABLATION_ARM_TO_ROW[arm]
        if not init_df.empty:
            sub = init_df[
                (init_df["task_set"] == task_set)
                & (init_df["milestone"] == milestone)
                & (init_df["n_proposals"] == target)
                & (init_df["arm"] == arm)
            ]
            if not sub.empty:
                by_row[row]["initialization"] = float(sub["mean_incumbent_mic"].iloc[0])
        if not final_df.empty:
            sub = final_df[
                (final_df["task_set"] == task_set)
                & (final_df["milestone"] == milestone)
                & (final_df["bo_calls"] == budget)
                & (final_df["target_pool_size"] == target)
                & (final_df["arm"] == arm)
            ]
            if not sub.empty:
                by_row[row]["final_bo"] = float(sub["mean_best_mic"].iloc[0])

    return [by_row[row] for row in ABLATION_ROW_ORDER]


def to_latex(rows: list[dict]) -> str:
    lines = [
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


def generate(
    results_dirs: str | Path | list[str | Path],
    task_set: str,
    milestone: int,
    out_dir: Path,
    target_pool_size: int | None = None,
    oracle_budget: int | None = None,
) -> Path | None:
    rows = build_rows(results_dirs, task_set, milestone, target_pool_size, oracle_budget)
    if all(r["initialization"] is None and r["final_bo"] is None for r in rows):
        print("[eval2.ablation] no data at all, skipping")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f"ablation_{DOMAIN_SLUG}.csv", index=False)
    tex_path = out_dir / f"ablation_{DOMAIN_SLUG}.tex"
    tex_path.write_text(to_latex(rows) + "\n")
    print(f"[eval2.ablation] wrote {tex_path} (and matching .csv)")
    return tex_path
