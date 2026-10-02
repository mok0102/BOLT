"""Numerically verify f_t^* for every Branin task, per impl_plan/motivational_exp.txt:

    "Do not assume the standard Branin optimum is unchanged when t changes.
    Numerically verify the optimum for each t using a high-accuracy global /
    multi-start procedure and save the resulting x_t^* and f_t^*."

Every existing regret computation in this repo (synthetic_experiment/
aggregate.py::build_heldout_per_task, eval2/table_baselines.py::compute,
benchmark_branin_bo_init.py::run_trial) hard-codes ``optimum = 10.0 * task_t``
with no verification anywhere. This module is the verification the spec
requires -- it does not assume that formula, even though it is the thing
being checked.

Method: scipy.optimize.differential_evolution (a global optimizer, not a
local one seeded by a guess) over the real BRANIN_BOUNDS box, polished by
L-BFGS-B from its best point. Reuses BraninTask.raw (the actual minimized
Branin value the oracle/BO code evaluates) directly -- not a re-derivation
of the formula -- so this can never silently drift from what the rest of the
repo computes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

BOLT_ROOT = Path(__file__).resolve().parents[3]
if str(BOLT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOLT_ROOT))

from synthetic_experiment.branin import BRANIN_BOUNDS, BraninTask  # noqa: E402

# The repo-wide shortcut this module exists to check, not assume.
ANALYTIC_OPTIMUM_FORMULA = "10.0 * task_t"


@dataclass(frozen=True)
class VerifiedOptimum:
    task_t: float
    f_star: float
    x_star: tuple[float, float]
    analytic_10t: float
    matches_analytic: bool
    n_restarts_agreeing: int
    n_restarts_total: int


def verify_optimum(
    task_t: float,
    *,
    seed: int = 0,
    de_seed_count: int = 8,
    polish_restarts: int = 20,
    agreement_atol: float = 1e-6,
) -> VerifiedOptimum:
    """Global search (differential_evolution, several independent seeds) +
    local polish (multi-start L-BFGS-B from the global search's own basin and
    from an independent uniform grid), so the reported f_t^* is not an
    artifact of one optimizer run's seed.
    """
    task = BraninTask(task_t=task_t, maximize=False)  # raw, minimize convention
    bounds = BRANIN_BOUNDS

    def f(x: np.ndarray) -> float:
        return float(task.raw(x))

    rng = np.random.default_rng(seed)
    candidates: list[tuple[float, np.ndarray]] = []

    for de_seed in range(de_seed_count):
        result = differential_evolution(
            f, bounds=bounds, seed=int(rng.integers(0, 2**31 - 1)),
            tol=1e-12, maxiter=2000, popsize=40, polish=True, mutation=(0.5, 1.5),
            recombination=0.9, updating="deferred", workers=1,
        )
        candidates.append((float(result.fun), np.asarray(result.x, dtype=float)))

    # Independent local polish from a uniform multi-start grid, so the
    # candidate set isn't entirely DE-derived.
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    starts = rng.uniform(lows, highs, size=(polish_restarts, 2))
    for start in starts:
        result = minimize(f, start, method="L-BFGS-B", bounds=bounds, tol=1e-14)
        if result.success:
            candidates.append((float(result.fun), np.asarray(result.x, dtype=float)))

    best_value = min(v for v, _ in candidates)
    agreeing = [x for v, x in candidates if abs(v - best_value) <= 1e-6]
    best_x = agreeing[0]
    # Final high-precision polish from the best point found.
    polished = minimize(f, best_x, method="Nelder-Mead",
                        options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20000})
    f_star = float(min(best_value, float(polished.fun)))
    x_star = polished.x if float(polished.fun) <= best_value else best_x

    analytic = 10.0 * task_t
    return VerifiedOptimum(
        task_t=task_t,
        f_star=f_star,
        x_star=(float(x_star[0]), float(x_star[1])),
        analytic_10t=analytic,
        matches_analytic=bool(abs(f_star - analytic) <= agreement_atol),
        n_restarts_agreeing=len(agreeing),
        n_restarts_total=len(candidates),
    )


def verify_many(task_values: list[float], **kwargs) -> list[VerifiedOptimum]:
    return [verify_optimum(t, **kwargs) for t in task_values]


def save_verified_optima(results: list[VerifiedOptimum], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "task_t": r.task_t, "f_star": r.f_star, "x_star": list(r.x_star),
            "analytic_10t": r.analytic_10t, "matches_analytic": r.matches_analytic,
            "max_abs_diff_from_analytic": abs(r.f_star - r.analytic_10t),
            "n_restarts_agreeing": r.n_restarts_agreeing, "n_restarts_total": r.n_restarts_total,
        }
        for r in results
    ]
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_verified_optima(path: Path) -> dict[float, VerifiedOptimum]:
    """Keyed by task_t for O(1) lookup from regret computations."""
    rows = json.loads(path.read_text())
    return {
        row["task_t"]: VerifiedOptimum(
            task_t=row["task_t"], f_star=row["f_star"], x_star=tuple(row["x_star"]),
            analytic_10t=row["analytic_10t"], matches_analytic=row["matches_analytic"],
            n_restarts_agreeing=row["n_restarts_agreeing"], n_restarts_total=row["n_restarts_total"],
        )
        for row in rows
    }
