"""Runs Experiment C's Random / Prior-best-reuse / Context-regression arms
as real held-out BO trajectories (20 tasks, init_size + oracle_budget rows
each), via synthetic_experiment.steps.run_bo's `initial_x` override -- the
same continuation mechanism every other arm in this repository uses, so
only the initialization differs between arms, per the doc's "CRITICAL
CONTROL" for Experiment C.

Random is already implemented by synthetic_experiment itself (uniform
initial_points when checkpoint=None) -- this module runs it through the
identical run_bo call for a consistent heldout/<arm> layout, not a
reimplementation.

Output layout mirrors synthetic_experiment's own heldout convention exactly:
    <run_dir>/heldout/Random/task_XXXX.csv
    <run_dir>/heldout/PriorBestReuse/task_XXXX.csv
    <run_dir>/heldout/ContextRegression/task_XXXX.csv
so core/regret.py's trajectory_path() (same `arm in {"STBO","LLAMBO"}` style
no-milestone convention) reads them with zero special-casing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
BOLT_ROOT = PKG_ROOT.parents[1]
for _p in (BOLT_ROOT, PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from synthetic_experiment.config import ExperimentConfig  # noqa: E402
from synthetic_experiment.steps import run_bo  # noqa: E402
from synthetic_experiment.task_splits import task_name  # noqa: E402

from core.initializer_baselines import (  # noqa: E402
    context_to_optimum_regression_init, prior_best_reuse_init,
)

NO_MILESTONE_ARMS = ("Random", "PriorBestReuse", "ContextRegression")


def run_arm(
    cfg: ExperimentConfig, arm: str, n_train_tasks: int, seed_offset: int = 200_000,
) -> list[Path]:
    """Evaluates `arm` on every held-out task, writing
    <run_dir>/heldout/<arm>/task_XXXX.csv. n_train_tasks is the training-task
    count PriorBestReuse/ContextRegression draw their cross-task signal from
    (T=50 for the doc's main Figure C; pass a smaller value to compare against
    an earlier milestone's worth of experience)."""
    if arm not in NO_MILESTONE_ARMS:
        raise ValueError(f"arm must be one of {NO_MILESTONE_ARMS}, got {arm!r}")
    out_dir = cfg.run_dir / "heldout" / arm
    written = []
    for index, task_t in enumerate(cfg.heldout_task_values):
        destination = out_dir / f"{task_name(index)}.csv"
        seed = cfg.bo_seed + seed_offset + index
        if arm == "Random":
            initial_x = None  # run_bo's own uniform _uniform_points(seed, init_size) path
        elif arm == "PriorBestReuse":
            initial_x = prior_best_reuse_init(cfg.trajectories_dir, n_train_tasks, cfg.init_size)
        else:
            initial_x = context_to_optimum_regression_init(
                cfg.trajectories_dir, n_train_tasks, task_t, cfg.init_size
            )
        written.append(run_bo(cfg, task_t, destination, seed=seed, initial_x=initial_x))
    return written


def run_all(cfg: ExperimentConfig, n_train_tasks: int) -> dict[str, list[Path]]:
    return {arm: run_arm(cfg, arm, n_train_tasks) for arm in NO_MILESTONE_ARMS}


if __name__ == "__main__":
    import argparse

    from synthetic_experiment.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-train-tasks", type=int, default=50)
    parser.add_argument("--arm", choices=[*NO_MILESTONE_ARMS, "all"], default="all")
    args = parser.parse_args()
    cfg = load_config(args.config)
    arms = NO_MILESTONE_ARMS if args.arm == "all" else [args.arm]
    for arm in arms:
        paths = run_arm(cfg, arm, args.n_train_tasks)
        print(f"{arm}: wrote {len(paths)} trajectories under {cfg.run_dir / 'heldout' / arm}")
