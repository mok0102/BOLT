"""Reference-aligned candidate distribution q_t and background sampling
(imp_plan/06_orpt_matched_intervention_plan.md, appendix.tex eq.
app-reference-candidate-distribution): q_t(x) ~ exp(log pi_ref(x|C) / tau_q)
over a fold's E_pool, tau_q=1.0 by default (decision 4 -- log
entropy/effective-sample-size diagnostics, no tuning loop).
"""

from __future__ import annotations

import math
import random

from .candidate_bank import EligibleCandidate


class ReferenceAlignedDistribution:
    def __init__(self, candidates: list[EligibleCandidate], log_likelihoods: dict[str, float], tau_q: float):
        assert len(candidates) >= 1
        self.candidates = candidates
        self.tau_q = tau_q
        weights = [math.exp(log_likelihoods[c.seq] / tau_q) for c in candidates]
        total = sum(weights)
        self.probs = [w / total for w in weights]

    def diagnostics(self) -> dict[str, float]:
        """Weight entropy (nats) and effective sample size -- watch for
        collapse onto a tiny subset of candidates (decision 4)."""
        entropy = -sum(p * math.log(p) for p in self.probs if p > 0)
        ess = 1.0 / sum(p * p for p in self.probs)
        return {"entropy": entropy, "effective_sample_size": ess, "n_candidates": len(self.candidates)}

    def sample_background(self, m_minus_1: int, exclude_seqs: set[str], rng: random.Random) -> list[EligibleCandidate] | None:
        """Samples m-1 unique candidates without replacement from q_t,
        excluding exclude_seqs (the two intervention candidates). Returns
        None if the eligible support is too small."""
        eligible_idx = [i for i, c in enumerate(self.candidates) if c.seq not in exclude_seqs]
        if len(eligible_idx) < m_minus_1:
            return None

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
