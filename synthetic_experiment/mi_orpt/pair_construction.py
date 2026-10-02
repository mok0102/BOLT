from __future__ import annotations

import math
import random
from pathlib import Path

from .background_sampler import ReferenceAlignedDistribution
from .candidate_bank import EligibleCandidate
from .one_step_evaluator import run_candidates_one_step


def paired_difference_stats(diffs: list[float]) -> tuple[float, float]:
    mean = sum(diffs) / len(diffs)
    if len(diffs) < 2:
        return mean, float("inf")
    se = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (len(diffs) * (len(diffs) - 1)))
    return mean, se


def construct_pairs_for_task(cfg, task, bank: list[EligibleCandidate],
                             log_likelihoods: dict[str, float], rng: random.Random, work_dir: Path) -> list[dict]:
    if len(bank) < 3:
        return []
    count = min(cfg.mi_max_candidates_per_task, len(bank) - 1)
    candidates = rng.sample(bank, count)
    reserved = {c.seq for c in candidates}
    q = ReferenceAlignedDistribution(bank, log_likelihoods, cfg.mi_tau_q)
    backgrounds, seeds = [], []
    for _ in range(cfg.mi_num_backgrounds):
        background = q.sample_background(cfg.init_size - 1, reserved, rng)
        if background is not None:
            backgrounds.append(background); seeds.append(rng.randrange(2**31 - 1))
    if len(backgrounds) < 2:
        return []
    utilities = run_candidates_one_step(cfg, task, backgrounds, candidates, work_dir, seeds)
    pairs = []
    for new in range(len(candidates)):
        for old in range(new):
            delta, se = paired_difference_stats([utilities[new][i] - utilities[old][i] for i in range(len(backgrounds))])
            z = abs(delta) / (se + 1e-12)
            if abs(delta) <= cfg.mi_delta_t or z < cfg.mi_z_min:
                continue
            chosen, rejected = (candidates[new], candidates[old]) if delta > 0 else (candidates[old], candidates[new])
            # The prompt asks for a good initial point. Even when a candidate
            # improves the subsequent BO acquisition, never teach DPO to
            # prefer it when its own objective score is worse.
            if chosen.y <= rejected.y:
                continue
            # task_split/task_index are the authoritative identity (orpt.py
            # rebuilds the DPO prompt from cfg.train_tasks[task_index], never
            # trusting a transform/descriptor baked into this diagnostics
            # record); reference_sequence is informational only.
            pairs.append({"task": task.name, "task_split": task.split, "task_index": task.index,
                          "reference_sequence": task.descriptor,
                          "chosen_sequence": chosen.seq, "chosen_score": chosen.y,
                          "rejected_sequence": rejected.seq, "rejected_score": rejected.y,
                          "delta": delta, "se": se})
    pairs.sort(key=lambda row: abs(row["delta"]) / (row["se"] + 1e-12), reverse=True)
    return pairs[:cfg.mi_target_pairs_per_task]
