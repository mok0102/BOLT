"""Numerically verify a Branin task's global optimum -- never assume it from
the transform formula, even though the transform's effect on the optimum is
analytically predictable (see ``branin.py::canonical_global_minimizers``).
Every existing regret computation in this repo that used to hard-code
``optimum = 10.0 * task_t`` had no verification at all; this module is the
verification that practice was missing, generalized from a per-``t``
scalar lookup to a per-task-transform search.

Lives here (not in ``experiments/eval2_branin_motivation/``) because the
task-manifest BUILDER (``task_splits.py``) needs this search to gate task
generation itself -- a shared-library concern, not an experiment-specific
one. The motivation package's own ``core/global_optimum.py`` re-exports
this module and adds an independent re-audit against the frozen manifest.

Method: scipy.optimize.differential_evolution (a global optimizer, not a
local one seeded by a guess) over the real BRANIN_BOUNDS box, polished by
L-BFGS-B from its best point, with a final Nelder-Mead polish -- unchanged
from the original scalar-``t`` version of this search, just applied to a
transformed task's objective instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .branin import BRANIN_BOUNDS, CANONICAL_F_STAR, BraninTask, canonical_global_minimizers


@dataclass(frozen=True)
class VerifiedOptimum:
    f_star: float
    x_star: tuple[float, float]  # the in-box minimizer with the largest edge margin (deterministic choice)
    x_star_all: tuple[tuple[float, float], ...]  # every in-box minimizer found, by edge margin descending
    edge_margin: float  # x_star's distance to the nearest box edge
    boundary_min: float  # diagnostic only (W3): min of f over the box's own edges, not gated on by default
    matches_canonical: bool
    n_restarts_agreeing: int
    n_restarts_total: int


def edge_margin(point: tuple[float, float], bounds=BRANIN_BOUNDS) -> float:
    """Distance from `point` to the nearest edge of `bounds` (negative if
    outside)."""
    (lo1, hi1), (lo2, hi2) = bounds
    return float(min(point[0] - lo1, hi1 - point[0], point[1] - lo2, hi2 - point[1]))


def _boundary_min(task: BraninTask, *, bounds=BRANIN_BOUNDS, n_per_edge: int = 200) -> float:
    """Diagnostic only (W3 in the task-generation design): the minimum of
    f over the box's own 4 edges. A minimizer sitting just outside the box
    makes this low without making f_star wrong -- recorded, not gated on by
    default (gating on it would reject most draws and reshape the sampled
    distribution for no correctness reason)."""
    (lo1, hi1), (lo2, hi2) = bounds
    s = np.linspace(0.0, 1.0, n_per_edge)
    edges = np.concatenate([
        np.stack([np.full(n_per_edge, lo1), lo2 + s * (hi2 - lo2)], axis=-1),
        np.stack([np.full(n_per_edge, hi1), lo2 + s * (hi2 - lo2)], axis=-1),
        np.stack([lo1 + s * (hi1 - lo1), np.full(n_per_edge, lo2)], axis=-1),
        np.stack([lo1 + s * (hi1 - lo1), np.full(n_per_edge, hi2)], axis=-1),
    ])
    return float(np.min(task.raw(edges)))


def verify_optimum(
    task: BraninTask,
    *,
    seed: int = 0,
    de_seed_count: int = 8,
    polish_restarts: int = 20,
    agreement_atol: float = 1e-6,
) -> VerifiedOptimum:
    """Global search (differential_evolution, several independent seeds) +
    local polish (multi-start L-BFGS-B from the global search's own basin and
    from an independent uniform grid) + a final Nelder-Mead polish, so the
    reported f_star is not an artifact of one optimizer run's seed.
    """
    bounds = BRANIN_BOUNDS

    def f(x: np.ndarray) -> float:
        return float(task.raw(x))

    rng = np.random.default_rng(seed)
    candidates: list[tuple[float, np.ndarray]] = []

    for _ in range(de_seed_count):
        result = differential_evolution(
            f, bounds=bounds, seed=int(rng.integers(0, 2**31 - 1)),
            tol=1e-12, maxiter=2000, popsize=40, polish=True, mutation=(0.5, 1.5),
            recombination=0.9, updating="deferred", workers=1,
        )
        candidates.append((float(result.fun), np.asarray(result.x, dtype=float)))

    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    starts = rng.uniform(lows, highs, size=(polish_restarts, 2))
    for start in starts:
        result = minimize(f, start, method="L-BFGS-B", bounds=bounds, tol=1e-14)
        if result.success:
            candidates.append((float(result.fun), np.asarray(result.x, dtype=float)))

    best_value = min(v for v, _ in candidates)
    agreeing = [x for v, x in candidates if abs(v - best_value) <= agreement_atol]
    best_x = agreeing[0]
    polished = minimize(f, best_x, method="Nelder-Mead",
                        options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20000})
    # Nelder-Mead is unbounded -- a polished point drifting outside the box
    # would silently corrupt the margin/well-posedness checks downstream.
    polished_in_box = all(lo <= v <= hi for v, (lo, hi) in zip(polished.x, bounds))
    use_polished = polished_in_box and float(polished.fun) <= best_value
    f_star = float(polished.fun) if use_polished else best_value
    x_star_primary = (float(polished.x[0]), float(polished.x[1])) if use_polished else (float(best_x[0]), float(best_x[1]))

    # x_star_all: map every canonical minimizer through this task's own
    # transform, keep the ones that land in-box AND match the found f_star
    # (the analytically-predicted set, cross-checked against the numeric
    # search rather than trusted alone).
    candidate_images = [
        tuple(float(v) for v in task.transform.from_canonical(np.array(m)))
        for m in canonical_global_minimizers()
    ]
    in_box = [
        pt for pt in candidate_images
        if bounds[0][0] <= pt[0] <= bounds[0][1] and bounds[1][0] <= pt[1] <= bounds[1][1]
        and abs(f(np.array(pt)) - f_star) <= max(agreement_atol, 1e-4)
    ]
    if not in_box:
        # Numeric search found an in-box optimum that the analytic image
        # list didn't reproduce (e.g. polish landed off a true minimizer by
        # more than tolerance) -- fall back to the single numeric point
        # rather than silently reporting an empty x_star_all.
        in_box = [x_star_primary]
    in_box.sort(key=lambda pt: edge_margin(pt, bounds), reverse=True)
    x_star_all = tuple(in_box)
    x_star = x_star_all[0]

    return VerifiedOptimum(
        f_star=f_star,
        x_star=x_star,
        x_star_all=x_star_all,
        edge_margin=edge_margin(x_star, bounds),
        boundary_min=_boundary_min(task, bounds=bounds),
        matches_canonical=bool(abs(f_star - CANONICAL_F_STAR) <= agreement_atol),
        n_restarts_agreeing=len(agreeing),
        n_restarts_total=len(candidates),
    )


def is_well_posed(verified: VerifiedOptimum, *, agreement_atol: float = 1e-6, min_edge_margin: float = 1.0) -> bool:
    """The two acceptance gates a generated task must pass (task_splits.py's
    rejection-sampling loop):

    W1 -- the verified in-box minimum matches the canonical value within
    tolerance (rejects a task whose true minimum sits on/outside the
    boundary, where the search can't have found the real unconstrained
    optimum).
    W2 -- at least one verified global minimizer is >= min_edge_margin from
    every box edge (deliberately "at least one", not "all": canonical
    Branin's own 3*pi minimizer sits only ~0.575 units from an edge, so
    requiring every minimizer to clear a 1.0-unit margin would make most
    draws unsatisfiable for reasons unrelated to well-posedness).
    """
    return (
        verified.matches_canonical
        and abs(verified.f_star - CANONICAL_F_STAR) <= agreement_atol
        and verified.edge_margin >= min_edge_margin
    )
