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

pair_construction.py's cache-and-reuse design (memos/suggestion.txt)
evaluates one candidate against a *shared, fixed* set of M backgrounds at
a time (run_candidate_one_step below), rather than evaluating two
candidates (an "arm" pair) against M *independently redrawn per pair*
backgrounds -- reusing the same M evaluations across every pairwise
comparison derived from them instead of paying fresh evaluations per
comparison.
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


def run_candidate_one_step(
    cfg: ExperimentConfig,
    task_idx: int,
    backgrounds: list[list[EligibleCandidate]],
    candidate: EligibleCandidate,
    cand_work_dir: Path,
    seeds: list[int],
    worker_pool: ProcessPoolExecutor | None = None,
) -> list[float]:
    """Evaluates `candidate` against every given (fixed, shared-across-the-
    whole-task) background, returning [U1(A_r + candidate) for r in
    backgrounds] in the same order as `backgrounds`. Each background
    evaluation is fully independent real-BO work.

    worker_pool=None (default): runs run_one_step_pool serially, one call
    at a time on cfg.cuda_visible_devices.

    worker_pool given (mi_orpt/warm_pool.py's create_pool()): dispatches
    all M evaluations concurrently across the pool -- no per-item GPU
    pinning needed here, each worker already permanently owns one GPU and
    has the LOLBO import stack warmed up (see warm_pool.py)."""
    background_seqs_ys = [([c.seq for c in bg], [c.y for c in bg]) for bg in backgrounds]
    work_dirs = [cand_work_dir / f"bg_{bg_idx:03d}" for bg_idx in range(len(backgrounds))]

    if worker_pool is None:
        return [
            run_one_step_pool(
                cfg,
                task_idx,
                bg_seqs + [candidate.seq],
                bg_ys + [candidate.y],
                work_dir,
                run_id=f"mi-onestep-{'-'.join(work_dir.parts[-2:])}",
                seed=seed,
            )
            for (bg_seqs, bg_ys), work_dir, seed in zip(background_seqs_ys, work_dirs, seeds)
        ]

    futures: dict[Future, int] = {}
    for bg_idx, ((bg_seqs, bg_ys), work_dir, seed) in enumerate(zip(background_seqs_ys, work_dirs, seeds)):
        payload = {
            "cfg": cfg,
            "task_idx": task_idx,
            "pool_seqs": bg_seqs + [candidate.seq],
            "pool_ys": bg_ys + [candidate.y],
            "work_dir": work_dir,
            "run_id": f"mi-onestep-{'-'.join(work_dir.parts[-2:])}",
            "seed": seed,
        }
        futures[worker_pool.submit(_run_one_in_worker, payload)] = bg_idx

    results: list[float | None] = [None] * len(backgrounds)
    for future in as_completed(futures):
        bg_idx = futures[future]
        ok, value = future.result()
        if not ok:
            raise RuntimeError(f"one-step BO call failed (task {task_idx}, bg {bg_idx}, candidate {candidate.seq!r}): {value}")
        results[bg_idx] = value
    return results


def zero_step_utility(background: list[EligibleCandidate], candidate: EligibleCandidate) -> float:
    """The zero-step ablation (experiments.tex sec:ablations): the pool's
    own best already-known value, no BO round, zero additional oracle cost."""
    return max([c.y for c in background] + [candidate.y])
