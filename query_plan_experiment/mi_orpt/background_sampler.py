"""Stable reference-likelihood weights and unique matched backgrounds."""

from __future__ import annotations

import math
import random

from .candidate_bank import EligibleCandidate


class ReferenceAlignedDistribution:
    def __init__(self, candidates: list[EligibleCandidate], log_likelihoods: dict[str, float], tau_q: float):
        assert len(candidates) >= 1
        self.candidates = candidates
        self.tau_q = tau_q
        log_weights = [log_likelihoods[c.seq] / tau_q for c in candidates]
        if not all(math.isfinite(w) for w in log_weights):
            raise ValueError("Nonfinite reference likelihood")
        offset = max(log_weights)
        weights = [math.exp(max(w - offset, -700)) for w in log_weights]
        total = sum(weights)
        self.probs = [w / total for w in weights]

    def diagnostics(self) -> dict[str, float]:
        """Weight entropy (nats) and effective sample size -- watch for
        collapse onto a tiny subset of candidates (decision 4)."""
        entropy = -sum(p * math.log(p) for p in self.probs if p > 0)
        ess = 1.0 / sum(p * p for p in self.probs)
        return {"entropy": entropy, "effective_sample_size": ess, "n_candidates": len(self.candidates)}

    def sample_background(self, m_minus_1: int, exclude_seqs: set[str], rng: random.Random, allow_replacement: bool = False) -> list[EligibleCandidate] | None:
        """Samples m-1 unique candidates without replacement from q_t,
        excluding exclude_seqs (the two intervention candidates). Returns
        None if the eligible support is too small."""
        eligible_idx = [i for i, c in enumerate(self.candidates) if c.seq not in exclude_seqs]
        if len(eligible_idx) < m_minus_1:
            if not allow_replacement or not eligible_idx:
                return None
            weights = [self.probs[i] for i in eligible_idx]
            # Preserve every available background plan, then pad with weighted
            # resampling. Repeated rows retain their original measured scores.
            selected = list(eligible_idx)
            selected += rng.choices(eligible_idx, weights=weights, k=m_minus_1-len(selected))
            return [self.candidates[i] for i in selected]

        remaining_idx = list(eligible_idx)
        remaining_weights = [self.probs[i] for i in remaining_idx]
        chosen: list[EligibleCandidate] = []
        for _ in range(m_minus_1):
            total = sum(remaining_weights)
            r = rng.random() * total
            cumulative = 0.0
            pick_pos = len(remaining_idx) - 1
            for pos, w in enumerate(remaining_weights):
                cumulative += w
                if r <= cumulative:
                    pick_pos = pos
                    break
            chosen.append(self.candidates[remaining_idx[pick_pos]])
            del remaining_idx[pick_pos]
            del remaining_weights[pick_pos]
        return chosen
