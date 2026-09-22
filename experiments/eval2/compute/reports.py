"""Text reports on the two baselines whose results don't fit the main plots.

POGPE/SGPE overload the milestone field to mean n_experts, so they must never
be drawn against the real milestone axis -- they get a table instead.

LLAMBO can stop early when it exhausts its input-token budget before reaching
the oracle budget, which makes its numbers not directly comparable to the
other arms. That truncation is surfaced explicitly rather than hidden.

Both were an inline heredoc in the old eval runbook; they are a module here
so the shell script stays a launcher and the logic stays testable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.results_io import load_summary_fixed_target_bo


def _load_per_task(results_dirs: list[str | Path]) -> pd.DataFrame:
    frames = []
    for d in results_dirs:
        path = Path(d) / "per_task_fixed_target_bo.csv"
        if not path.exists():
            print(f"[eval2.reports] {path} not found, skipping")
            continue
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def gp_expert_report(summary: pd.DataFrame) -> None:
    print("\n=== POGPE/SGPE: mean_best_mic by n_experts (own axis, not milestone) ===")
    gpe = summary[summary["arm"].isin(["POGPE", "SGPE"])].sort_values(["arm", "milestone"])
    if gpe.empty:
        print("(no rows)")
        return
    table = gpe[["arm", "milestone", "n_tasks_ran_bo", "mean_best_mic"]].rename(columns={"milestone": "n_experts"})
    print(table.to_string(index=False))


def llambo_report(per_task: pd.DataFrame, bo_calls: int) -> None:
    print("\n=== LLAMBO: token-budget early-termination report ===")
    if per_task.empty or "llambo_terminated_early" not in per_task.columns:
        print("(no LLAMBO rows with recorded llambo_terminated_early)")
        return
    llambo = per_task[per_task["arm"] == "LLAMBO"].dropna(subset=["llambo_terminated_early"])
    if llambo.empty:
        print("(no LLAMBO rows with recorded llambo_terminated_early)")
        return
    n = len(llambo)
    n_early = int(llambo["llambo_terminated_early"].sum())
    print(f"{n_early}/{n} tasks ({100 * n_early / n:.1f}%) terminated early due to the "
          f"llambo_max_input_tokens budget, before reaching oracle_budget={bo_calls} real BO calls")
    print(llambo["llambo_input_tokens_used"].describe().to_string())


def run(results_dirs: list[str | Path], bo_calls: int) -> None:
    summary = load_summary_fixed_target_bo(results_dirs)
    if summary.empty:
        print("[eval2.reports] no summary_fixed_target_bo.csv anywhere, nothing to report")
        return
    summary = summary[summary["bo_calls"] == bo_calls]
    gp_expert_report(summary)
    llambo_report(_load_per_task(results_dirs), bo_calls)
