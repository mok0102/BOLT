"""Actual one-step BO evaluator for mol's matched-intervention ORPT. Mirrors
peptide_experiment/mi_orpt/one_step_evaluator.py's shape and algorithm exactly
(read-only reference, never imported) -- domain-generic pool -> real-BO-round ->
U_1 logic, only run_bo swapped for mol_run_bo.

worker_pool (mi_orpt/warm_pool.py) is a ProcessPoolExecutor, one worker
process per GPU, each permanently pinned to its own device -- when given,
every (candidate, background) cell is dispatched to it instead of running
serially on this process's own single device. Unlike peptide's own
warm_pool.py, mol's payoff is pure GPU parallelism, not import-cost
amortization: mol_run_bo (mol_experiment/steps.py) already runs in-process,
so there's no subprocess-per-call tax to amortize the way peptide's
subprocess-based run_bo paid.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mol_experiment.config import MolExperimentConfig  # noqa: E402
from mol_experiment.steps import mol_run_bo  # noqa: E402
from .candidate_bank import EligibleCandidate  # noqa: E402


def run_one_step_pool(
    cfg: MolExperimentConfig,
    task_idx: int,
    pool_seqs: list[str],
    pool_ys: list[float],
    work_dir: Path,
    run_id: str,
    seed: int,
) -> float:
    """Writes pool_seqs/pool_ys as init files (already-known trajectory values,
    never re-scored), runs exactly one real acquisition round (oracle_budget =
    cfg.bsz), and returns U_1 = max(train_y) over the resulting trajectory."""
    work_dir.mkdir(parents=True, exist_ok=True)
    init_path = work_dir / f"task_{task_idx:04d}_init.txt"
    scores_path = work_dir / f"task_{task_idx:04d}_scores.csv"
    init_path.write_text("\n".join(pool_seqs) + "\n")
    scores_path.write_text("\n".join(f"{y:.8f}" for y in pool_ys) + "\n")

    run_cfg = dataclasses.replace(cfg, oracle_budget=cfg.bsz, init_size=len(pool_seqs))
    csv_path = mol_run_bo(
        run_cfg, task_idx, work_dir, run_id=run_id, init_path=init_path, scores_path=scores_path,
        n_bo_steps=1, seed=seed,
    )
    df = pd.read_csv(csv_path)
    return float(df["train_y"].max())


def run_candidates_one_step(
    cfg: MolExperimentConfig,
    task_idx: int,
    backgrounds: list[list[EligibleCandidate]],
    candidates: list[EligibleCandidate],
    work_dir_root: Path,
    seeds: list[int],
    worker_pool=None,
) -> list[list[float]]:
    """Evaluates every candidate against every given (fixed, shared-across-the-
    whole-task) background, returning result[c][r] = U1(backgrounds[r] +
    candidates[c]). worker_pool=None (default): serial, on this process's own
    device. A mi_orpt/warm_pool.py ProcessPoolExecutor: every (candidate,
    background) cell is submitted independently and reassembled in the same
    [c][r] shape -- order doesn't matter for correctness (each cell is an
    independent one-step-BO call), only for matching the return shape."""
    background_seqs_ys = [([c.seq for c in bg], [c.y for c in bg]) for bg in backgrounds]

    if worker_pool is None:
        return [
            [
                run_one_step_pool(
                    cfg,
                    task_idx,
                    bg_seqs + [candidate.seq],
                    bg_ys + [candidate.y],
                    work_dir_root / f"cand_{c_idx:04d}" / f"bg_{bg_idx:03d}",
                    run_id=f"mi-onestep-cand_{c_idx:04d}-bg_{bg_idx:03d}",
                    seed=seed,
                )
                for bg_idx, ((bg_seqs, bg_ys), seed) in enumerate(zip(background_seqs_ys, seeds))
            ]
            for c_idx, candidate in enumerate(candidates)
        ]

    from .warm_pool import _run_one_step_in_worker

    futures = {}
    for c_idx, candidate in enumerate(candidates):
        for bg_idx, ((bg_seqs, bg_ys), seed) in enumerate(zip(background_seqs_ys, seeds)):
            payload = {
                "cfg": cfg,
                "task_idx": task_idx,
                "pool_seqs": bg_seqs + [candidate.seq],
                "pool_ys": bg_ys + [candidate.y],
                "work_dir": work_dir_root / f"cand_{c_idx:04d}" / f"bg_{bg_idx:03d}",
                "run_id": f"mi-onestep-cand_{c_idx:04d}-bg_{bg_idx:03d}",
                "seed": seed,
            }
            futures[worker_pool.submit(_run_one_step_in_worker, payload)] = (c_idx, bg_idx)

    results: list[list[float | None]] = [[None] * len(backgrounds) for _ in candidates]
    for future in futures:
        c_idx, bg_idx = futures[future]
        ok, value = future.result()
        if not ok:
            raise RuntimeError(f"one-step BO worker failed (candidate {c_idx}, background {bg_idx}): {value}")
        results[c_idx][bg_idx] = value
    return results


def zero_step_utility(candidate: EligibleCandidate) -> float:
    """Byte-identical logic to peptide's own zero_step_utility -- the candidate's
    own already-known objective value, no shared base set, no BO round."""
    return candidate.y
