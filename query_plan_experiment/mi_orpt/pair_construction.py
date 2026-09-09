"""Compare candidates on shared reference-aligned backgrounds.

Each reserved candidate is evaluated once per background, then all pairwise
utility differences reuse those results. Keep at most target_pairs passing
the paired mean/SE reliability criterion. Serial DB dispatch is intentional.
"""

from __future__ import annotations

import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..config import ExperimentConfig
from .background_sampler import ReferenceAlignedDistribution
from .candidate_bank import EligibleCandidate, min_bank_size_needed
from .one_step_evaluator import run_candidates_one_step, zero_step_utility


def propose_candidates(bank: list[EligibleCandidate], k: int, rng: random.Random) -> list[EligibleCandidate]:
    """Uniform random k distinct candidates from the task's eligible bank
    -- no objective-rank stratification (appendix.tex never specifies a
    selection heuristic beyond drawing intervention candidates from E_t).
    Raw objective value never determines a pair's label; that's decided
    later, only by sign(Delta_1)."""
    return rng.sample(bank, min(k, len(bank)))


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
    max_candidates: int,
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

    Reserves up to max_candidates distinct candidates upfront and excludes
    the whole reserved set from every one of the num_backgrounds shared
    backgrounds (so every (candidate, background) combination is valid by
    construction -- no candidate ever ends up inside a background it's
    later compared against). Evaluates every reserved candidate against
    every background in a single batch (each candidate costs exactly
    num_backgrounds real-BO calls per bo_steps=1, or the free zero-step
    ablation per bo_steps=0) -- no early stopping, so the run always costs
    exactly len(candidate_pool)*num_backgrounds real-BO calls regardless
    of how many pairs end up passing the reliability filter. Every pair of
    evaluated candidates is then compared using the cached per-background
    utilities, no fresh evaluation needed per comparison. If fewer than
    target_pairs comparisons pass the reliability filter, returns however
    many did (no forcing). If more pass, keeps only the target_pairs
    comparisons with the highest |z| (most statistically confident), so
    the output size stays an exact, interpretable control on the high end.

    worker_pool: an optional warm-worker pool (mi_orpt/warm_pool.py's
    create_pool(), created once per build_pairs.py invocation and shared
    across every task in that milestone) forwarded straight through to
    run_candidates_one_step, which submits every candidate's every
    background evaluation to it in one batch -- letting multiple
    candidates' one-step-BO calls run concurrently across the whole pool.
    None falls back to serial dispatch."""
    min_needed = min_bank_size_needed(m, num_reserved=max_candidates)
    if len(bank) < 3:
        print(f"[mi_orpt pair_construction] task ref={reference_sequence!r}: bank={len(bank)} (need >=3 distinct plans), skipping")
        return []

    q_t = ReferenceAlignedDistribution(bank, log_likelihoods, tau_q)

    candidate_count = min(max_candidates, len(bank)-1)
    padded = len(bank) < min_needed
    if padded:
        print(f"[MI padding] bank={len(bank)}, reserved={candidate_count}, "
              f"background size={m-1}: repeat existing background plans as needed")
    candidate_pool = propose_candidates(bank, candidate_count, rng)
    if len(candidate_pool) < 2:
        print(f"[mi_orpt pair_construction] task ref={reference_sequence!r}: only {len(candidate_pool)} candidate(s) reserved, skipping")
        return []

    # Sample all M backgrounds (+ matched seeds) once, excluding the whole
    # reserved candidate pool -- this is the only part that touches the
    # shared rng, so it must stay in a fixed, deterministic order
    # regardless of whether the real-BO calls below run serially or in
    # parallel.
    reserved_seqs = {c.seq for c in candidate_pool}
    backgrounds: list[list[EligibleCandidate]] = []
    seeds: list[int] = []
    for _ in range(num_backgrounds):
        background = q_t.sample_background(m - 1, exclude_seqs=reserved_seqs, rng=rng, allow_replacement=True)
        if background is None:
            break
        backgrounds.append(background)
        seeds.append(rng.randrange(2**31 - 1))

    if len(backgrounds) < 2:
        print(f"[mi_orpt pair_construction] task ref={reference_sequence!r}: only {len(backgrounds)} usable background(s), skipping")
        return []

    if bo_steps == 1:
        u1 = run_candidates_one_step(cfg, task_idx, backgrounds, candidate_pool, work_dir_root, seeds, worker_pool=worker_pool)
    else:
        assert bo_steps == 0, bo_steps
        u1 = [[zero_step_utility(background, candidate) for background in backgrounds] for candidate in candidate_pool]

    kept_pairs: list[dict] = []
    for new_idx in range(len(candidate_pool)):
        for old_idx in range(new_idx):
            diffs = [u1[new_idx][r] - u1[old_idx][r] for r in range(len(backgrounds))]
            delta, se = paired_difference_stats(diffs)
            z = abs(delta) / (se + 1e-12)
            print(
                f"[mi_orpt pair_construction] candidates ({new_idx},{old_idx}): diffs={diffs} "
                f"delta={delta:.4f} se={se:.4f} z={z:.3f} (need |delta|>{delta_t} and z>={z_min})"
            )
            if abs(delta) <= delta_t or z < z_min:
                continue

            x_new, x_old = candidate_pool[new_idx], candidate_pool[old_idx]
            chosen, rejected = (x_new, x_old) if delta > 0 else (x_old, x_new)
            kept_pairs.append(
                {
                    "reference_sequence": reference_sequence,
                    "chosen_sequence": chosen.seq,
                    "chosen_score": chosen.y,
                    "rejected_sequence": rejected.seq,
                    "rejected_score": rejected.y,
                    "delta": delta,
                    "se": se,
                    "background_padding": padded,
                    "unique_bank_size": len(bank),
                    "reserved_candidates": candidate_count,
                }
            )

    if len(kept_pairs) > target_pairs:
        kept_pairs.sort(key=lambda p: abs(p["delta"]) / (p["se"] + 1e-12), reverse=True)
        kept_pairs = kept_pairs[:target_pairs]

    print(
        f"[mi_orpt pair_construction] task ref={reference_sequence!r}: "
        f"kept {len(kept_pairs)}/{target_pairs} target after {len(candidate_pool)} candidates evaluated (cap {max_candidates})"
    )
    return kept_pairs
