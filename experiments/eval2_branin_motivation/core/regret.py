"""Simple-regret aggregation for Experiment A/C, reading f_t^* from the
VERIFIED optimum file (core/global_optimum.py) rather than re-deriving or
re-hardcoding ``10*task_t`` the way synthetic_experiment/aggregate.py and
eval2/table_baselines.py both do.

    simple_regret_b(t) = f_t(x_best_after_b_calls) - f_t^*

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

from synthetic_experiment.task_splits import task_name  # noqa: E402
from core.global_optimum import load_verified_optima  # noqa: E402

VERIFIED_OPTIMA_HELDOUT = Path(__file__).resolve().parents[1] / "results" / "verified_optima_heldout.json"
VERIFIED_OPTIMA_TRAIN = Path(__file__).resolve().parents[1] / "results" / "verified_optima_train.json"


def f_star_lookup(task_t: float, *, atol: float = 1e-9) -> float:
    """f_t^* for a task_t value, read from whichever verified-optima file has
    it (train or heldout) -- raises if the task_t wasn't verified, rather
    than falling back to the unverified 10*t formula (F1/F6-style: no silent
    substitution)."""
    for path in (VERIFIED_OPTIMA_HELDOUT, VERIFIED_OPTIMA_TRAIN):
        table = load_verified_optima(path)
        for t, verified in table.items():
            if abs(t - task_t) <= atol:
                return verified.f_star
    raise KeyError(
        f"task_t={task_t!r} has no verified global optimum in {VERIFIED_OPTIMA_HELDOUT} or "
        f"{VERIFIED_OPTIMA_TRAIN} -- run core/global_optimum.py first; this module never "
        "falls back to the unverified 10*task_t shortcut."
    )


def trajectory_path(run_dir: Path, arm: str, milestone: int | None, task_index: int) -> Path:
    # milestone is None, not a hardcoded {"STBO", "LLAMBO"} set, is the real
    # criterion for "no milestone suffix in the directory name" -- it also
    # covers core/run_initializer_baselines.py's Random/PriorBestReuse/
    # ContextRegression arms, which are likewise milestone-independent and
    # write to heldout/<arm>/ directly (see that module's own docstring).
    directory = arm if milestone is None else f"{arm}-{milestone}"
    return run_dir / "heldout" / directory / f"{task_name(task_index)}.csv"


def regret_curve(
    run_dir: Path, arm: str, milestone: int | None, task_index: int, task_t: float,
    init_size: int, oracle_budget: int,
) -> pd.Series:
    """Simple regret at b=0..oracle_budget, indexed by b. Best-so-far is a
    running cummax of train_y (BraninTask's maximize-convention score),
    mirroring synthetic_experiment/eval2/data.py::task_curves exactly (same
    cummax-then-slice logic) so this is a drop-in alternative to that module
    for everything except the regret-vs-raw-score framing."""
    path = trajectory_path(run_dir, arm, milestone, task_index)
    frame = pd.read_csv(path)
    if "train_y" not in frame or len(frame) < init_size + oracle_budget:
        raise ValueError(f"{path}: expected >= {init_size + oracle_budget} rows with train_y")
    if not (frame.task_t.iloc[0] == task_t or abs(float(frame.task_t.iloc[0]) - task_t) < 1e-9):
        raise ValueError(f"{path}: task_t {frame.task_t.iloc[0]} != expected {task_t}")
    f_star = f_star_lookup(task_t)
    best_score = frame.train_y.astype(float).cummax().iloc[init_size - 1: init_size + oracle_budget]
    best_value = -best_score.reset_index(drop=True)  # train_y = -f_t(x); best_value = f_t(x_best)
    regret = (best_value - f_star).clip(lower=0.0)
    regret.index.name = "b"
    return regret


def regret_table(
    run_dir: Path, arm: str, milestone: int | None, heldout_task_values: list[float],
    init_size: int, oracle_budget: int, checkpoints: list[int],
) -> pd.DataFrame:
    """One row per (task_index, b) for b in `checkpoints`, for a single arm
    at a single milestone -- the per-task raw data Figure A's panels and
    Figure C's trajectory plot are built from. Always kept alongside any
    aggregated mean, per the doc's 'store per-task, per-seed raw results
    before aggregation.'"""
    rows = []
    for task_index, task_t in enumerate(heldout_task_values):
        curve = regret_curve(run_dir, arm, milestone, task_index, task_t, init_size, oracle_budget)
        for b in checkpoints:
            if b not in curve.index:
                raise ValueError(f"b={b} not in regret curve index (0..{oracle_budget}) for {arm}-{milestone}")
            rows.append({
                "arm": arm, "milestone": milestone, "task_index": task_index, "task_t": task_t,
                "b": b, "simple_regret": float(curve.loc[b]),
            })
    return pd.DataFrame(rows)


def full_trajectory_table(
    run_dir: Path, arm: str, milestone: int | None, heldout_task_values: list[float],
    init_size: int, oracle_budget: int,
) -> pd.DataFrame:
    """Every b=0..oracle_budget, every held-out task -- the full regret
    trajectory the doc's secondary Figure-A plot and Figure C both need."""
    rows = []
    for task_index, task_t in enumerate(heldout_task_values):
        curve = regret_curve(run_dir, arm, milestone, task_index, task_t, init_size, oracle_budget)
        for b, regret in curve.items():
            rows.append({
                "arm": arm, "milestone": milestone, "task_index": task_index, "task_t": task_t,
                "b": int(b), "simple_regret": float(regret),
            })
    return pd.DataFrame(rows)
