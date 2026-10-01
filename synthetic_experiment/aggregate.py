from __future__ import annotations

import pandas as pd

from .config import ExperimentConfig
from .task_splits import task_name


def arms(cfg: ExperimentConfig) -> list[str]:
    kinds = ["BOLT"]
    if cfg.build_orpt:
        kinds.append("ORPT")
    if cfg.build_mtbo:
        kinds.append("MTBO")
    if cfg.build_pogpe:
        kinds.append("POGPE")
    if cfg.build_sgpe:
        kinds.append("SGPE")
    if cfg.build_optformer:
        kinds.append("OptFormer")
    result = [f"{kind}-{m}" for kind in kinds for m in cfg.milestones]
    if cfg.build_stbo:
        result.append("STBO")
    if cfg.build_llambo:
        result.append("LLAMBO")
    return result


def build_heldout_per_task(cfg: ExperimentConfig) -> pd.DataFrame:
    """Write plotting-ready heldout results for every arm, t, and budget k.

    BraninTask returns -f_t because BOLT maximizes scores.  For this task
    family the exact minimum is f_t*=10t, so simple_regret=f_best-10t and
    zero is optimal for every t.
    """

    checkpoints = sorted(set(cfg.table_k_checkpoints or [cfg.oracle_budget]))
    records = []
    for arm in arms(cfg):
        method, _, milestone_text = arm.partition("-")
        for index, task_t in enumerate(cfg.heldout_task_values):
            path = cfg.heldout_dir / arm / f"{task_name(index)}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            for k in checkpoints:
                window = frame.iloc[: cfg.init_size + k]
                if len(window) < cfg.init_size + k:
                    continue
                best_score = float(window.train_y.max())
                best_value = -best_score
                optimum_value = 10.0 * task_t
                records.append({
                    "arm": arm,
                    "method": method,
                    "milestone": int(milestone_text) if milestone_text else None,
                    "task_index": index,
                    "task_t": task_t,
                    "oracle_calls": k,
                    "best_score": best_score,
                    "best_value": best_value,
                    "optimum_value": optimum_value,
                    "simple_regret": max(best_value - optimum_value, 0.0),
                    "trajectory_path": str(path),
                })
    result = pd.DataFrame(records)
    cfg.aggregate_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(cfg.aggregate_dir / "heldout_per_task.csv", index=False)
    return result


def build_summary(cfg: ExperimentConfig) -> pd.DataFrame:
    per_task = build_heldout_per_task(cfg)
    records = []
    checkpoints = sorted(set(cfg.table_k_checkpoints or [cfg.oracle_budget]))
    for arm in arms(cfg):
        for k in checkpoints:
            values = per_task[(per_task.arm == arm) & (per_task.oracle_calls == k)]
            records.append({
                "arm": arm,
                "oracle_calls": k,
                "mean_best_score": values.best_score.mean() if not values.empty else None,
                "mean_simple_regret": values.simple_regret.mean() if not values.empty else None,
                "num_tasks": len(values),
            })
    result = pd.DataFrame(records)
    cfg.aggregate_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(cfg.aggregate_dir / "summary.csv", index=False)
    return result
