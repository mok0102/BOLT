"""Builds a Table-1-style CSV from held-out eval outputs. See
imp_plan/02_query_plan_reimplementation_plan.md /
imp_plan/03_experiment_results_guide.md for how to read it.

Arms are derived from cfg.milestones rather than hardcoded, so this also
works unmodified against a smoke config's tiny milestones.

Best-score-at-k EXCLUDES censored (timed-out/failed) rows before taking the
max -- direct analogy to peptide_experiment/aggregate.py's constraint-
infeasibility filter: a censored row's train_y is a lower-bound proxy score
(see your_objective_functions.py::DatabaseObjective), not a genuine completed
runtime, so it could make a timed-out query look like an arm's "best" result.
A window with zero completed rows returns None (missing), same as peptide's
precedent -- never falls back to a censored value.
"""

from __future__ import annotations

import pandas as pd

from .config import ExperimentConfig


def _arms(cfg: ExperimentConfig) -> list[str]:
    return [f"{kind}-{m}" for kind in (["BOLT", "ORPT"] if cfg.build_orpt else ["BOLT"]) for m in cfg.milestones]


def best_runtime_at_k(csv_path, init_size: int, k: int) -> float | None:
    """Best (lowest) completed runtime among the first init_size+k logged
    rows. train_y is already -runtime (maximization framing); reported here
    as a positive runtime (lower = better), mirroring peptide's own
    MIC = -train_y convention."""
    df = pd.read_csv(csv_path)
    row_idx = min(init_size + k, len(df))
    if row_idx <= 0 or df.empty:
        return None
    window = df.iloc[:row_idx]
    uncensored = window[window["censoring"].astype(float) == 0.0]
    if uncensored.empty:
        return None
    best_y = uncensored["train_y"].max()
    return -best_y


def _sum_runtime_across_tasks(cfg: ExperimentConfig, arm: str, k: int) -> tuple[float, int]:
    arm_dir = cfg.heldout_dir / arm
    total = 0.0
    n_missing = 0
    for workload in cfg.heldout_tasks:
        csv_path = arm_dir / f"{workload}.csv"
        if not csv_path.exists():
            n_missing += 1
            continue
        runtime = best_runtime_at_k(csv_path, cfg.init_size, k)
        if runtime is not None:
            total += runtime
        else:
            n_missing += 1
    return total, n_missing


def build_table1(cfg: ExperimentConfig) -> pd.DataFrame:
    """Table 1: rows = oracle-call checkpoints, columns = BOLT-<m> arms,
    cells = summed best-completed-runtime across the 99-task held-out set."""
    arms = _arms(cfg)
    n_tasks = len(cfg.heldout_tasks)
    checkpoints = sorted(set(cfg.table_k_checkpoints or [cfg.oracle_budget]))
    rows = {}
    for k in checkpoints:
        row = {}
        for arm in arms:
            total, n_missing = _sum_runtime_across_tasks(cfg, arm, k)
            if n_missing:
                print(f"[table1] arm={arm} k={k}: missing {n_missing}/{n_tasks} task CSVs")
            row[arm] = total
        rows[k] = row
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "oracle_calls"

    cfg.aggregate_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.aggregate_dir / "table1.csv"
    df.to_csv(out_path)
    print(f"Wrote {out_path}")
    return df
