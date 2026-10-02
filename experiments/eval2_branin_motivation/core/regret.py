"""Simple-regret aggregation for Experiment A/C, reading f_t^* directly from
a task's own TaskRecord (synthetic_experiment/task_splits.py), which already
carries its numerically verified optimum from the frozen task manifest --
never assumed from a formula, the way synthetic_experiment/aggregate.py and
eval2/table_baselines.py both still do (and would now do *wrongly*, since
every task's canonical f_star is the same constant under the affine-
transform family, not ``10*task_t``).

    simple_regret_b(t) = f(x_best_after_b_calls) - f_t^*

b counts BO oracle calls AFTER initialization (b=0 means immediately after
evaluating the init pool), per impl_plan/motivational_exp.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BOLT_ROOT = Path(__file__).resolve().parents[3]
if str(BOLT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOLT_ROOT))

from synthetic_experiment.task_splits import TaskRecord  # noqa: E402


def trajectory_path(run_dir: Path, arm: str, milestone: int | None, task: TaskRecord) -> Path:
    # milestone is None, not a hardcoded {"STBO", "LLAMBO"} set, is the real
    # criterion for "no milestone suffix in the directory name" -- it also
    # covers core/run_initializer_baselines.py's Random/PriorBestReuse/
    # ContextRegression arms, which are likewise milestone-independent and
    # write to heldout/<arm>/ directly (see that module's own docstring).
    directory = arm if milestone is None else f"{arm}-{milestone}"
    return run_dir / "heldout" / directory / f"{task.name}.csv"


def regret_curve(
    run_dir: Path, arm: str, milestone: int | None, task: TaskRecord,
    init_size: int, oracle_budget: int,
) -> pd.Series:
    """Simple regret at b=0..oracle_budget, indexed by b. Best-so-far is a
    running cummax of train_y (BraninTask's maximize-convention score),
    mirroring synthetic_experiment/eval2/data.py::task_curves exactly (same
    cummax-then-slice logic) so this is a drop-in alternative to that module
    for everything except the regret-vs-raw-score framing."""
    path = trajectory_path(run_dir, arm, milestone, task)
    frame = pd.read_csv(path, dtype={"task_split": str})
    if "train_y" not in frame or len(frame) < init_size + oracle_budget:
        raise ValueError(f"{path}: expected >= {init_size + oracle_budget} rows with train_y")
    if frame.task_split.iloc[0] != task.split or int(frame.task_index.iloc[0]) != task.index:
        raise ValueError(
            f"{path}: trajectory task identity ({frame.task_split.iloc[0]}, "
            f"{frame.task_index.iloc[0]}) != expected ({task.split}, {task.index})"
        )
    f_star = task.verified.f_star
    best_score = frame.train_y.astype(float).cummax().iloc[init_size - 1: init_size + oracle_budget]
    best_value = -best_score.reset_index(drop=True)  # train_y = -f_t(x); best_value = f_t(x_best)
    regret = (best_value - f_star).clip(lower=0.0)
    regret.index.name = "b"
    return regret


def regret_table(
    run_dir: Path, arm: str, milestone: int | None, tasks: tuple[TaskRecord, ...],
    init_size: int, oracle_budget: int, checkpoints: list[int],
) -> pd.DataFrame:
    """One row per (task_index, b) for b in `checkpoints`, for a single arm
    at a single milestone -- the per-task raw data Figure A's panels and
    Figure C's trajectory plot are built from. Always kept alongside any
    aggregated mean, per the doc's 'store per-task, per-seed raw results
    before aggregation.'"""
    rows = []
    for task in tasks:
        curve = regret_curve(run_dir, arm, milestone, task, init_size, oracle_budget)
        for b in checkpoints:
            if b not in curve.index:
                raise ValueError(f"b={b} not in regret curve index (0..{oracle_budget}) for {arm}-{milestone}")
            rows.append({
                "arm": arm, "milestone": milestone, "task_index": task.index,
                "b": b, "simple_regret": float(curve.loc[b]),
            })
    return pd.DataFrame(rows)


def full_trajectory_table(
    run_dir: Path, arm: str, milestone: int | None, tasks: tuple[TaskRecord, ...],
    init_size: int, oracle_budget: int,
) -> pd.DataFrame:
    """Every b=0..oracle_budget, every held-out task -- the full regret
    trajectory the doc's secondary Figure-A plot and Figure C both need."""
    rows = []
    for task in tasks:
        curve = regret_curve(run_dir, arm, milestone, task, init_size, oracle_budget)
        for b, regret in curve.items():
            rows.append({
                "arm": arm, "milestone": milestone, "task_index": task.index,
                "b": int(b), "simple_regret": float(regret),
            })
    return pd.DataFrame(rows)
