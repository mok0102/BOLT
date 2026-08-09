"""Ties together reference-aligned background sampling and the actual
one-step BO evaluator (or the zero-step ablation) into reliability-filtered
candidate-level preference pairs (paper/method.tex sec:one-step-pool-evaluation,
paper/appendix.tex app:pair-construction).
"""

from __future__ import annotations

import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..config import ExperimentConfig
from .background_sampler import ReferenceAlignedDistribution
from .candidate_bank import EligibleCandidate, min_bank_size_needed
from .one_step_evaluator import run_matched_pairs_one_step, zero_step_utility


def propose_pairs(bank: list[EligibleCandidate], k: int, rng: random.Random) -> list[tuple[EligibleCandidate, EligibleCandidate]]:
    """Uniform random distinct pairs from the task's eligible bank -- no
    objective-rank stratification, no hard-negative sampling (appendix.tex
    never specifies a pairing heuristic beyond drawing intervention pairs
    from E_t). Raw objective value never determines the label; that's
    decided later, only by sign(Delta_1)."""
    if len(bank) < 2:
        return []
    pairs = []
    seen = set()
    max_attempts = max(k * 20, 200)
    attempts = 0
    while len(pairs) < k and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(bank, 2)
        key = frozenset({a.seq, b.seq})
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b))
    return pairs


def paired_difference_stats(diffs: list[float]) -> tuple[float, float]:
    """Delta_1 and its Monte Carlo standard error over M shared backgrounds
    (method.tex eq. paired-mc-difference/paired-mc-se)."""
    m = len(diffs)
    mean = sum(diffs) / m
    if m < 2:
        return mean, float("inf")
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (m * (m - 1)))
    return mean, se


def construct_pairs_for_task(
    cfg: ExperimentConfig,
    task_idx: int,
    bank: list[EligibleCandidate],
    reference_sequence: str,
    log_likelihoods: dict[str, float],
    m: int,
    target_pairs: int,
    max_attempts: int,
    num_backgrounds: int,
    tau_q: float,
    z_min: float,
    delta_t: float,
    bo_steps: int,
    work_dir_root: Path,
    rng: random.Random,
    worker_pool: ProcessPoolExecutor | None = None,
) -> list[dict]:
    """Returns kept preference-pair records
    {reference_sequence, chosen_sequence, chosen_score, rejected_sequence,
    rejected_score, delta, se} -- same reference_sequence/*_sequence/
    *_score fields make_dpo_train_data_csv.py's pairs carry, so
    build_pairs.py can reuse its make_messages()/write_jsonl() unchanged.

    Keeps proposing and testing new candidate pairs (real one-step BO calls
    per bo_steps=1, or the free zero-step ablation per bo_steps=0) until
    either target_pairs reliable pairs are found, or max_attempts candidates
    have been tried -- unlike objective_ranked's sample_pairs(), the
    reliability filter can reject any given candidate, so a fixed proposal
    count would leave the actual training-pair yield unknown from the
    config alone (see config.py's mi_target_pairs_per_task docstring).

    worker_pool: an optional warm-worker pool (mi_orpt/warm_pool.py's
    create_pool(), created once per build_pairs.py invocation and shared
    across every task in that milestone) forwarded straight through to
    run_matched_pairs_one_step; None falls back to serial dispatch."""
    kept_pairs: list[dict] = []

    min_needed = min_bank_size_needed(m)
    if len(bank) < min_needed:
        print(f"[mi_orpt pair_construction] task ref={reference_sequence!r}: bank={len(bank)} (need >={min_needed}), skipping")
        return kept_pairs

    q_t = ReferenceAlignedDistribution(bank, log_likelihoods, tau_q)

    attempts = 0
    for pair_idx, (x_i, x_j) in enumerate(propose_pairs(bank, max_attempts, rng)):
        if len(kept_pairs) >= target_pairs:
            break
        attempts += 1

        # Sample all backgrounds (+ their matched seeds) serially first --
        # this is the only part that touches the shared rng, so it must
        # stay in a fixed, deterministic order regardless of whether the
        # actual BO calls below run serially or in parallel.
        backgrounds = []
        seeds = []
        for _ in range(num_backgrounds):
            background = q_t.sample_background(m - 1, exclude_seqs={x_i.seq, x_j.seq}, rng=rng)
            if background is None:
                break
            backgrounds.append(background)
            seeds.append(rng.randrange(2**31 - 1))

        if bo_steps == 1:
            pair_work_dir = work_dir_root / f"pair_{pair_idx:04d}"
            diffs = run_matched_pairs_one_step(cfg, task_idx, backgrounds, x_i, x_j, pair_work_dir, seeds, worker_pool=worker_pool)
        else:
            assert bo_steps == 0, bo_steps
            diffs = [zero_step_utility(background, x_i) - zero_step_utility(background, x_j) for background in backgrounds]

        if len(diffs) < 2:
            continue

        delta, se = paired_difference_stats(diffs)
        z = abs(delta) / (se + 1e-12)
        print(f"[mi_orpt pair_construction] pair {pair_idx}: diffs={diffs} delta={delta:.4f} se={se:.4f} z={z:.3f} (need |delta|>{delta_t} and z>={z_min})")
        if abs(delta) <= delta_t:
            continue
        if z < z_min:
            continue

        chosen, rejected = (x_i, x_j) if delta > 0 else (x_j, x_i)
        kept_pairs.append(
            {
                "reference_sequence": reference_sequence,
                "chosen_sequence": chosen.seq,
                "chosen_score": chosen.y,
                "rejected_sequence": rejected.seq,
                "rejected_score": rejected.y,
                "delta": delta,
                "se": se,
            }
        )

    print(
        f"[mi_orpt pair_construction] task ref={reference_sequence!r}: "
        f"kept {len(kept_pairs)}/{target_pairs} target after {attempts} attempts (cap {max_attempts})"
    )
    return kept_pairs
