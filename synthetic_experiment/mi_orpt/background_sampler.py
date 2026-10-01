from __future__ import annotations

import math
import random

from .candidate_bank import EligibleCandidate


class ReferenceAlignedDistribution:
    def __init__(self, candidates: list[EligibleCandidate], log_likelihoods: dict[str, float], tau: float):
        self.candidates = candidates
        logits = [log_likelihoods[c.seq] / tau for c in candidates]
        offset = max(logits)
        weights = [math.exp(max(v - offset, -700.0)) for v in logits]
        total = sum(weights)
        self.probs = [v / total for v in weights]

    def sample_background(self, size: int, excluded: set[str], rng: random.Random) -> list[EligibleCandidate] | None:
        indices = [i for i, c in enumerate(self.candidates) if c.seq not in excluded]
        if not indices:
            return None
        weights = [self.probs[i] for i in indices]
        if len(indices) >= size:
            selected = []
            for _ in range(size):
                pos = rng.choices(range(len(indices)), weights=weights, k=1)[0]
                selected.append(indices.pop(pos)); weights.pop(pos)
        else:
            selected = list(indices) + rng.choices(indices, weights=weights, k=size - len(indices))
        return [self.candidates[i] for i in selected]
