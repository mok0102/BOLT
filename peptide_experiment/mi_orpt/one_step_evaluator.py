"""Actual one-step BO evaluator for matched-intervention ORPT
(paper/method.tex sec:outcome-utility, paper/appendix.tex
app:one-step-evaluator-details).

For a matched pool S = A ∪ {x}, fits a fresh surrogate on S's already-known
scores (no re-scoring -- values come from the completed trajectory) and
runs exactly one real BO acquisition round -- the deployment acquisition
rule, over the deployment search space, at deployment batch size -- via
the existing steps.run_bo() subprocess (real oracle calls, not a lookup
against a logged bank; no discrete future-acquisition candidate bank is
needed at all, per appendix.tex app:candidate-banks). U_1(C,S) = max(score
over S plus that one batch).

Also supports the zero-step ablation (experiments.tex sec:ablations,
"matched zero-step pool-outcome tuning"): rank by the pool's own
best-already-known value, no BO round, zero additional oracle cost.
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ..config import ExperimentConfig
from ..steps import run_bo
from .candidate_bank import EligibleCandidate
from .warm_pool import _run_one_in_worker


def run_one_step_pool(
    cfg: ExperimentConfig,
    task_idx: int,
    pool_seqs: list[str],
    pool_ys: list[float],
    work_dir: Path,
    run_id: str,
    seed: int,
) -> float:
    """Writes pool_seqs/pool_ys as init files (already-known trajectory
    values, never re-scored), runs exactly one real acquisition round
    (oracle_budget = cfg.bsz, deployment's own batch size) via the
    unmodified deployment BO path, and returns U_1 = max(train_y) over the
    resulting pool+one-batch trajectory."""
    work_dir.mkdir(parents=True, exist_ok=True)
    init_path = work_dir / f"task_{task_idx:04d}_init.txt"
    scores_path = work_dir / f"task_{task_idx:04d}_scores.csv"
    init_path.write_text("\n".join(pool_seqs) + "\n")
    scores_path.write_text("\n".join(f"{y:.8f}" for y in pool_ys) + "\n")

    run_cfg = dataclasses.replace(cfg, oracle_budget=cfg.bsz, init_size=len(pool_seqs))
    csv_path = run_bo(run_cfg, task_idx, work_dir, run_id=run_id, init_path=init_path, scores_path=scores_path, seed=seed)
    df = pd.read_csv(csv_path)
    return float(df["train_y"].max())


def run_matched_pair_one_step(
    cfg: ExperimentConfig,
    task_idx: int,
    background: list[EligibleCandidate],
    x_i: EligibleCandidate,
    x_j: EligibleCandidate,
    work_dir_root: Path,
    seed: int,
) -> tuple[float, float]:
    """Runs the one-step evaluator for S_i=A+[x_i] and S_j=A+[x_j], unique
    work dirs per arm (steps.run_bo()'s dest_csv is keyed by task_idx only
    within a work_dir, so distinct arms need distinct dirs). Same seed for
    both arms: matches surrogate/acquisition random state as far as
    steps.run_bo()'s existing --seed hook allows; oracle noise that can't
    be coupled is treated as ordinary outcome variability (appendix.tex
    app:noisy-one-step-evaluator).

    run_id is derived from work_dir_root's own unique path components (not
    just the arm label) because steps.run_bo() stages its raw LOLBO output
    at a *shared* path keyed only by {experiment_id}_{run_id}_task_{task_idx}
    (optimization_all_collected_data/) before copying it into work_dir --
    two concurrent calls for the same task_idx+label (run_matched_pairs_one_step's
    parallel dispatch) would otherwise race on that shared filename."""
    background_seqs = [c.seq for c in background]
    background_ys = [c.y for c in background]
    run_id_suffix = "-".join(work_dir_root.parts[-2:]) if len(work_dir_root.parts) >= 2 else work_dir_root.name

    u1 = {}
    for label, candidate in (("i", x_i), ("j", x_j)):
        pool_seqs = background_seqs + [candidate.seq]
        pool_ys = background_ys + [candidate.y]
        work_dir = work_dir_root / label
        run_id = f"mi-onestep-{label}-{run_id_suffix}"
        u1[label] = run_one_step_pool(cfg, task_idx, pool_seqs, pool_ys, work_dir, run_id, seed)

    return u1["i"], u1["j"]


def run_matched_pairs_one_step(
    cfg: ExperimentConfig,
    task_idx: int,
    backgrounds: list[list[EligibleCandidate]],
    x_i: EligibleCandidate,
    x_j: EligibleCandidate,
    pair_work_dir: Path,
    seeds: list[int],
    worker_pool: ProcessPoolExecutor | None = None,
) -> list[float]:
    """Returns [u1_i - u1_j, ...] in the same order as `backgrounds`. Each
    background (and its two arms) is fully independent real-BO work.

    worker_pool=None (default): runs run_matched_pair_one_step serially,
    one call at a time on cfg.cuda_visible_devices -- unchanged fallback
    behavior for configs without mi_parallel_gpus set.

    worker_pool given (mi_orpt/warm_pool.py's create_pool(), created once
    per build_pairs.py invocation and shared across every task/pair in
    that milestone): flattens to one work item per (background, arm) --
    2x per background -- and dispatches all of them to the pool at once.
    No per-item GPU pinning needed here (unlike the old thread+subprocess
    dispatch this replaced): each worker already permanently owns one GPU,
    claimed once at pool-creation time, and -- critically -- already has
    the whole LOLBO import stack warmed up, so per-item cost is just the
    real work (~2s), not ~13s of Python imports repeated every call."""
    assert len(backgrounds) == len(seeds)
    work_dirs = [pair_work_dir / f"bg_{bg_idx:03d}" for bg_idx in range(len(backgrounds))]

    if worker_pool is None:
        diffs = []
        for background, work_dir, seed in zip(backgrounds, work_dirs, seeds):
            u1_i, u1_j = run_matched_pair_one_step(cfg, task_idx, background, x_i, x_j, work_dir, seed)
            diffs.append(u1_i - u1_j)
        return diffs

    futures: dict[Future, tuple[int, str]] = {}
    for bg_idx, (background, work_dir, seed) in enumerate(zip(backgrounds, work_dirs, seeds)):
        background_seqs = [c.seq for c in background]
        background_ys = [c.y for c in background]
        for label, candidate in (("i", x_i), ("j", x_j)):
            arm_work_dir = work_dir / label
            payload = {
                "cfg": cfg,
                "task_idx": task_idx,
                "pool_seqs": background_seqs + [candidate.seq],
                "pool_ys": background_ys + [candidate.y],
                "work_dir": arm_work_dir,
                "run_id": f"mi-onestep-{label}-{'-'.join(work_dir.parts[-2:])}",
                "seed": seed,
            }
            future = worker_pool.submit(_run_one_in_worker, payload)
            futures[future] = (bg_idx, label)

    results: dict[tuple[int, str], float] = {}
    for future in as_completed(futures):
        bg_idx, label = futures[future]
        ok, value = future.result()
        if not ok:
            raise RuntimeError(f"one-step BO call failed (task {task_idx}, bg {bg_idx}, arm {label}): {value}")
        results[(bg_idx, label)] = value

    return [results[(bg_idx, "i")] - results[(bg_idx, "j")] for bg_idx in range(len(backgrounds))]


def zero_step_utility(background: list[EligibleCandidate], candidate: EligibleCandidate) -> float:
    """The zero-step ablation (experiments.tex sec:ablations): the pool's
    own best already-known value, no BO round, no additional oracle cost."""
    return max([c.y for c in background] + [candidate.y])
