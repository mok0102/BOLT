"""Experiment C's two non-learned/simply-learned initializer baselines that
do not exist anywhere in this repository -- "keep implementation
deliberately simple and document it" (impl_plan/motivational_exp.txt).

Both produce an `init_size`-point initialization pool for a held-out task;
everything downstream (the BO continuation) reuses
synthetic_experiment.steps.run_bo unmodified via its `initial_x` override --
the same mechanism the real BOLT/ORPT sampling path uses -- so only the
*initialization* differs between every Experiment-C arm, per that
experiment's "CRITICAL CONTROL" section.

A documented nuance for motivation_results.md, found while writing this:
this Branin task family's global-optimum x-location does NOT depend on
task_t (core/global_optimum.py's own verification: x* in {-pi, pi, 3*pi} for
every t in [0,1], since t only scales the additive cos(x1) term, never the
quadratic term that pins the minimizer's location). A trivial
"always propose the same point" strategy can therefore look deceptively
strong on this specific benchmark -- prior-best reuse is not a fair proxy
for "does cross-task transfer help" in general, only on a task family
structured this way. Flagged here, not hidden.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BOLT_ROOT = Path(__file__).resolve().parents[3]
if str(BOLT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOLT_ROOT))

from synthetic_experiment.branin import BRANIN_BOUNDS  # noqa: E402
from synthetic_experiment.task_splits import task_name  # noqa: E402

_LOWS = np.array([b[0] for b in BRANIN_BOUNDS])
_HIGHS = np.array([b[1] for b in BRANIN_BOUNDS])


def _load_train_points(trajectories_dir: Path, n_tasks: int) -> pd.DataFrame:
    """All (task_index, task_t, x1, x2, y) rows from the first n_tasks
    completed training trajectories (init + BO-acquired points alike --
    every evaluated point is fair game for a task-independent reuse rule,
    exactly mirroring how BOLT's own SFT dataset draws from the full
    trajectory, not just its own init pool)."""
    rows = []
    for index in range(n_tasks):
        path = trajectories_dir / f"{task_name(index)}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing training trajectory for task {index}: {path}")
        frame = pd.read_csv(path)
        for row in frame.itertuples():
            x = eval(row.train_x)  # noqa: S307 -- own trusted CSV, a JSON-ish "[a, b]" literal
            rows.append({"task_index": index, "task_t": row.task_t, "x1": x[0], "x2": x[1], "y": row.train_y})
    return pd.DataFrame(rows)


def prior_best_reuse_init(trajectories_dir: Path, n_train_tasks: int, init_size: int) -> np.ndarray:
    """Task-independent rule: the `init_size` distinct points with the
    highest observed score across every completed training task so far,
    reused VERBATIM for every held-out task regardless of its own task_t.
    No task-conditioning at all -- this is the simplest possible
    "cross-task transfer", a floor every learned initializer should beat."""
    points = _load_train_points(trajectories_dir, n_train_tasks)
    ranked = points.sort_values("y", ascending=False)
    seen: set[tuple[float, float]] = set()
    chosen: list[tuple[float, float]] = []
    for row in ranked.itertuples():
        key = (round(row.x1, 6), round(row.x2, 6))
        if key in seen:
            continue
        seen.add(key)
        chosen.append((row.x1, row.x2))
        if len(chosen) == init_size:
            break
    if len(chosen) < init_size:
        raise ValueError(
            f"prior_best_reuse_init: only {len(chosen)} distinct points available across "
            f"{n_train_tasks} training tasks, need {init_size}"
        )
    return np.asarray(chosen, dtype=float)


def context_to_optimum_regression_init(
    trajectories_dir: Path, n_train_tasks: int, task_t: float, init_size: int, *, k: int = 3,
) -> np.ndarray:
    """Simple non-LLM learned initializer: k-nearest-neighbor regression from
    task context (task_t, a scalar) to that task's own best observed point,
    fit on the n_train_tasks completed training tasks. Deliberately the
    simplest model that could plausibly work (doc: "keep it simple"), not a
    neural net -- this tests whether the (low-dimensional, scalar-context)
    Branin task can just be solved by direct cross-task regression, which is
    exactly the question Experiment C item 3 asks.

    Every one of the init_size proposed points is the SAME k-NN prediction
    (this baseline has one scalar input and no stochasticity, so there is
    nothing to diversify across the pool without inventing an arbitrary
    perturbation rule) -- kept simple and explicit, not papered over.
    """
    points = _load_train_points(trajectories_dir, n_train_tasks)
    best_per_task = points.loc[points.groupby("task_index").y.idxmax()]
    train_t = best_per_task.task_t.to_numpy(dtype=float)
    train_xy = best_per_task[["x1", "x2"]].to_numpy(dtype=float)

    k_eff = min(k, len(train_t))
    distances = np.abs(train_t - task_t)
    nearest = np.argsort(distances)[:k_eff]
    weights = 1.0 / (distances[nearest] + 1e-9)
    weights /= weights.sum()
    predicted = (weights[:, None] * train_xy[nearest]).sum(axis=0)
    predicted = np.clip(predicted, _LOWS, _HIGHS)
    return np.tile(predicted, (init_size, 1))
