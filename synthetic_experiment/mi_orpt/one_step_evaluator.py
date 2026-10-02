from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..steps import run_bo


def zero_step_utility(background, candidate) -> float:
    return max(c.y for c in [*background, candidate])


def _evaluate_one(cfg, task, background, candidate, root, seed):
    pool = [*background, candidate]
    mi_pool_size = cfg.mi_bo_candidate_pool_size or cfg.bo_candidate_pool_size
    context = {
        "task_split": task.split, "task_index": task.index, "task_manifest": cfg.manifest.token,
        "pool": [c.seq for c in pool],
        "seed": seed,
        "version": 6,  # v5 -> v6: cache key now hashes full task identity, not a bare task_t float
        "bo_steps": cfg.mi_bo_steps,
        "utility_mode": cfg.mi_utility_mode,
        "bo_candidate_pool_size": mi_pool_size,
        "bo_lengthscale": cfg.mi_bo_lengthscale,
        "bo_ucb_beta": cfg.mi_bo_ucb_beta,
    }
    key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()[:20]
    work = root / key
    path = work / "trajectory.csv"
    initial_x = np.asarray([c.x for c in pool])
    run_cfg = replace(
        cfg,
        init_size=len(pool),
        # The config requires a positive budget; max_bo_steps controls this evaluation.
        oracle_budget=max(1, cfg.mi_bo_steps),
        bo_candidate_pool_size=mi_pool_size,
        bo_lengthscale=cfg.mi_bo_lengthscale,
        bo_ucb_beta=cfg.mi_bo_ucb_beta,
    )
    run_bo(run_cfg, task, path, seed=seed, initial_x=initial_x, max_bo_steps=cfg.mi_bo_steps)
    frame = pd.read_csv(path)
    acquired_scores = frame.train_y.iloc[len(pool):]
    acquired_score = float(acquired_scores.iloc[-1]) if len(acquired_scores) else None
    terminal_best = max([candidate.y, *(float(y) for y in acquired_scores)])
    background_best = max(c.y for c in background)
    if cfg.mi_bo_steps == 0:
        utility = candidate.y if cfg.mi_utility_mode == "terminal_best" else float(frame.train_y.max())
    elif cfg.mi_utility_mode == "acquired_score":
        utility = acquired_score
    elif cfg.mi_utility_mode == "candidate_acquired_best":
        # Count the intervention itself as an observation. Do not include the
        # shared background best here, since it can mask candidate differences.
        utility = max(candidate.y, acquired_score)
    elif cfg.mi_utility_mode == "improvement":
        utility = acquired_score - background_best
    else:
        utility = terminal_best
    work.mkdir(parents=True, exist_ok=True)
    (work / "evaluation.json").write_text(json.dumps({
        **context,
        "utility_mode": cfg.mi_utility_mode,
        "utility": utility,
        "candidate_score": candidate.y,
        "acquired_score": acquired_score,
        "background_best": background_best,
        "terminal_best": terminal_best,
    }, indent=2))
    return utility


def run_candidates_one_step(cfg, task, backgrounds, candidates, root: Path, seeds):
    """Evaluate all candidate/background interventions concurrently.

    Jobs retain their preassigned shared seed and are written back to the
    original [candidate][background] location, so parallel completion order
    cannot change paired-difference statistics.
    """

    values = [[None] * len(backgrounds) for _ in candidates]
    jobs = [
        (candidate_idx, background_idx, candidate, background, seed)
        for candidate_idx, candidate in enumerate(candidates)
        for background_idx, (background, seed) in enumerate(zip(backgrounds, seeds))
    ]
    workers = min(cfg.mi_parallel_workers, len(jobs))
    print(f"[MI] running {len(jobs)} one-step BO jobs with {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mi-bo") as executor:
        futures = {
            executor.submit(_evaluate_one, cfg, task, background, candidate, root, seed):
            (candidate_idx, background_idx)
            for candidate_idx, background_idx, candidate, background, seed in jobs
        }
        completed = 0
        for future in as_completed(futures):
            candidate_idx, background_idx = futures[future]
            values[candidate_idx][background_idx] = future.result()
            completed += 1
            if completed == len(jobs) or completed % workers == 0:
                print(f"[MI] completed {completed}/{len(jobs)} one-step BO jobs", flush=True)
    return values
