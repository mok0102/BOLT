"""Ties together reference-aligned background sampling and the actual one-step BO
evaluator (or the zero-step ablation) into reliability-filtered candidate-level
preference pairs. Byte-identical algorithm to
peptide_experiment/mi_orpt/pair_construction.py (read-only reference, never
imported -- this logic is domain-generic: shared backgrounds, matched-pair
Delta_1/SE/z statistics, the H0 cumulative-agreement gate) -- only the
`reference_sequence` field is renamed to `task_context` (spec's "Task identity"
section: mol's context is a composite of protein sequence + seed SELFIES, not a
single reference string, so the field name should not imply otherwise) and
ExperimentConfig -> MolExperimentConfig.
"""

from __future__ import annotations

import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mol_experiment.config import MolExperimentConfig  # noqa: E402
from .background_sampler import ReferenceAlignedDistribution  # noqa: E402
from .candidate_bank import EligibleCandidate, min_bank_size_needed  # noqa: E402
from .one_step_evaluator import run_candidates_one_step, zero_step_utility  # noqa: E402


def propose_candidates(bank: list[EligibleCandidate], k: int, rng: random.Random) -> list[EligibleCandidate]:
    """Uniform random k distinct candidates from the task's eligible bank."""
    return rng.sample(bank, min(k, len(bank)))


def paired_difference_stats(diffs: list[float]) -> tuple[float, float]:
    """Delta_1 and its Monte Carlo standard error over M shared backgrounds."""
    m = len(diffs)
    mean = sum(diffs) / m
    if m < 2:
        return mean, float("inf")
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (m * (m - 1)))
    return mean, se


def construct_pairs_for_task(
    cfg: MolExperimentConfig,
    task_idx: int,
    bank: list[EligibleCandidate],
    task_context: str,
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
    require_h0: bool = True,
) -> list[dict]:
    """Returns kept preference-pair records {task_context, chosen_sequence,
    chosen_score, rejected_sequence, rejected_score, delta, se}. See
    peptide_experiment/mi_orpt/pair_construction.py::construct_pairs_for_task's
    docstring for the full cache-and-reuse / H0 rationale (unchanged here)."""
    min_needed = min_bank_size_needed(m, num_reserved=max_candidates)
    if len(bank) < min_needed:
        max_candidates = max(2, min(max_candidates, len(bank) - 2))
        m = max(2, len(bank) - max_candidates + 1)
        print(
            f"[mi_orpt pair_construction] task {task_idx}: bank={len(bank)} "
            f"(need >={min_needed} for the configured max_candidates/m), "
            f"using reduced max_candidates={max_candidates}/m={m} instead of skipping"
        )

    q_t = ReferenceAlignedDistribution(bank, log_likelihoods, tau_q)

    candidate_pool = propose_candidates(bank, max_candidates, rng)
    if len(candidate_pool) < 2:
        print(f"[mi_orpt pair_construction] task {task_idx}: only {len(candidate_pool)} candidate(s) reserved, skipping")
        return []

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
        print(f"[mi_orpt pair_construction] task {task_idx}: only {len(backgrounds)} usable background(s), skipping")
        return []

    if bo_steps == 1:
        u1 = run_candidates_one_step(cfg, task_idx, backgrounds, candidate_pool, work_dir_root, seeds, worker_pool=worker_pool)
    else:
        assert bo_steps == 0, bo_steps
        u1 = [[zero_step_utility(candidate)] * len(backgrounds) for candidate in candidate_pool]

    kept_pairs: list[dict] = []
    n_h0_rejected = 0
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
            if require_h0 and chosen.y <= rejected.y:
                n_h0_rejected += 1
                continue
            kept_pairs.append(
                {
                    "task_context": task_context,
                    "chosen_sequence": chosen.seq,
                    "chosen_score": chosen.y,
                    "rejected_sequence": rejected.seq,
                    "rejected_score": rejected.y,
                    "delta": delta,
                    "se": se,
                }
            )

    if len(kept_pairs) > target_pairs:
        kept_pairs.sort(key=lambda p: abs(p["delta"]) / (p["se"] + 1e-12), reverse=True)
        kept_pairs = kept_pairs[:target_pairs]

    print(
        f"[mi_orpt pair_construction] task {task_idx}: "
        f"kept {len(kept_pairs)}/{target_pairs} target after {len(candidate_pool)} candidates evaluated "
        f"(cap {max_candidates}), {n_h0_rejected} pair(s) rejected by the H0 necessary condition"
    )
    return kept_pairs
