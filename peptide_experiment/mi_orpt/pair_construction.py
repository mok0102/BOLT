"""Ties together reference-aligned background sampling and the actual
one-step BO evaluator (or the zero-step ablation) into reliability-filtered
candidate-level preference pairs (paper/method.tex sec:one-step-pool-evaluation,
paper/appendix.tex app:pair-construction).

Cache-and-reuse design (memos/suggestion.txt): rather than proposing
independent candidate pairs and paying 2*M fresh one-step-BO evaluations
per pair, this fixes M shared backgrounds once per task, evaluates each
candidate against them exactly once (M evaluations per candidate, cached),
and reads off every pairwise difference among already-evaluated candidates
for free. L evaluated candidates yield up to C(L,2) reliability-checked
comparisons from just M*L real-BO calls, instead of M*L*(L-1) under the
old per-pair design -- confirmed empirically this session (target=30 on a
real task needed 1072 calls under the old design; the same yield needs
roughly M*13~=104 under this one)."""

from __future__ import annotations

import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..config import ExperimentConfig
from .background_sampler import ReferenceAlignedDistribution
from .candidate_bank import EligibleCandidate, min_bank_size_needed
from .one_step_evaluator import run_candidate_one_step, zero_step_utility


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
    later compared against). Evaluates candidates one at a time (each
    costs exactly num_backgrounds real-BO calls per bo_steps=1, or the
    free zero-step ablation per bo_steps=0), caching each candidate's
    per-background utility; every newly-evaluated candidate is compared
    against every previously-evaluated one using the cached values, no
    fresh evaluation needed per comparison. Stops as soon as target_pairs
    reliable pairs are found, or max_candidates have been evaluated,
    whichever comes first. Growing by one candidate can unlock several
    passing comparisons at once, so the kept count can overshoot
    target_pairs in a single step -- when it does, keeps only the
    target_pairs comparisons with the highest |z| (most statistically
    confident), so the output size stays an exact, interpretable control
    regardless of how the stochastic growth landed.

    worker_pool: an optional warm-worker pool (mi_orpt/warm_pool.py's
    create_pool(), created once per build_pairs.py invocation and shared
    across every task in that milestone) forwarded straight through to
    run_candidate_one_step; None falls back to serial dispatch."""
    min_needed = min_bank_size_needed(m, num_reserved=max_candidates)
    if len(bank) < min_needed:
        print(f"[mi_orpt pair_construction] task ref={reference_sequence!r}: bank={len(bank)} (need >={min_needed}), skipping")
        return []

    q_t = ReferenceAlignedDistribution(bank, log_likelihoods, tau_q)

    candidate_pool = propose_candidates(bank, max_candidates, rng)
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
        background = q_t.sample_background(m - 1, exclude_seqs=reserved_seqs, rng=rng)
        if background is None:
            break
        backgrounds.append(background)
        seeds.append(rng.randrange(2**31 - 1))

    if len(backgrounds) < 2:
        print(f"[mi_orpt pair_construction] task ref={reference_sequence!r}: only {len(backgrounds)} usable background(s), skipping")
        return []

    u1_cache: dict[int, list[float]] = {}
    evaluated: list[int] = []
    kept_pairs: list[dict] = []

    for new_idx, candidate in enumerate(candidate_pool):
        if len(kept_pairs) >= target_pairs:
            break

        if bo_steps == 1:
            cand_work_dir = work_dir_root / f"cand_{new_idx:04d}"
            u1_cache[new_idx] = run_candidate_one_step(cfg, task_idx, backgrounds, candidate, cand_work_dir, seeds, worker_pool=worker_pool)
        else:
            assert bo_steps == 0, bo_steps
            u1_cache[new_idx] = [zero_step_utility(background, candidate) for background in backgrounds]

        for old_idx in evaluated:
            diffs = [u1_cache[new_idx][r] - u1_cache[old_idx][r] for r in range(len(backgrounds))]
            delta, se = paired_difference_stats(diffs)
            z = abs(delta) / (se + 1e-12)
            print(
                f"[mi_orpt pair_construction] candidates ({new_idx},{old_idx}): diffs={diffs} "
                f"delta={delta:.4f} se={se:.4f} z={z:.3f} (need |delta|>{delta_t} and z>={z_min})"
            )
            if abs(delta) <= delta_t or z < z_min:
                continue

            x_new, x_old = candidate, candidate_pool[old_idx]
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
                }
            )
        evaluated.append(new_idx)

    if len(kept_pairs) > target_pairs:
        kept_pairs.sort(key=lambda p: abs(p["delta"]) / (p["se"] + 1e-12), reverse=True)
        kept_pairs = kept_pairs[:target_pairs]

    print(
        f"[mi_orpt pair_construction] task ref={reference_sequence!r}: "
        f"kept {len(kept_pairs)}/{target_pairs} target after {len(evaluated)} candidates evaluated (cap {max_candidates})"
    )
    return kept_pairs
