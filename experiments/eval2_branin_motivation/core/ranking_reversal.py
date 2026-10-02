"""Experiment B: does standalone candidate quality agree with downstream
BO utility?

No existing script in this repository computes this -- confirmed while
planning (IMPLEMENTATION_PLAN.md section 5). Built from
synthetic_experiment/mi_orpt/'s existing primitives (candidate_bank,
background_sampler, one_step_evaluator, likelihood), which are reused
UNMODIFIED; this module only adds the horizon sweep and the reversal/
agreement statistics the doc asks for.

IMPORTANT, found while building this (a real confound, not a hypothetical
one): synthetic_experiment/mi_orpt/pair_construction.py's own pair
construction -- the one that fed real ORPT training -- filters out every
pair where `chosen.y <= rejected.y`, i.e. every case where the h-step-
preferred candidate does NOT also have the higher standalone score. That
is precisely a *reversal* in this experiment's terms, so the existing
orpt_pairs_*.diagnostics.json files (used to train real ORPT checkpoints)
are survivorship-biased against reversals by construction and CANNOT be
reused to measure a reversal rate. This module recomputes Delta_0/Delta_h
from scratch, before any such filter, and applies the same z-score
reliability criterion only as an explicit toggle (the doc's "report how
results change with and without reliability filtering").

Delta_0(i,j)   = y_i - y_j                              (standalone)
Delta_h(i,j)   = mean_r[ U_h(A_r + x_i) - U_h(A_r + x_j) ]  (matched, M backgrounds)

reversal_rate(h)      = P( sign(Delta_0) != sign(Delta_h) )
agreement_Htrain(h)   = P( sign(Delta_Htrain) == sign(Delta_h) ), Htrain=1

Near-ties (either Delta is ~0) are excluded from both statistics, not
counted as reversals/agreements either way -- "do not let tiny numerical
differences count as meaningful reversals."
"""

from __future__ import annotations

import dataclasses
import json
import random
import sys
from pathlib import Path

import numpy as np

BOLT_ROOT = Path(__file__).resolve().parents[3]
if str(BOLT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOLT_ROOT))

from synthetic_experiment.config import ExperimentConfig  # noqa: E402
from synthetic_experiment.mi_orpt.background_sampler import ReferenceAlignedDistribution  # noqa: E402
from synthetic_experiment.mi_orpt.candidate_bank import EligibleCandidate, build_eligible_bank  # noqa: E402
from synthetic_experiment.mi_orpt.likelihood import score_sequences  # noqa: E402
from synthetic_experiment.mi_orpt.one_step_evaluator import run_candidates_one_step  # noqa: E402
from synthetic_experiment.task_splits import task_name  # noqa: E402

HORIZONS = (1, 2, 3, 5, 10, 20, 50)
TRAIN_HORIZON = 1  # the H the main synthetic ORPT experiments train on

# Near-tie threshold on Delta_0: a standalone-score gap this small is not a
# meaningful preference to begin with, so a pair this close is excluded from
# BOTH the reversal and the agreement statistic rather than coin-flipped
# into either bucket. Independent of mi_z_min (that threshold is about
# Delta_h's own reliability, not Delta_0's).
DELTA0_TIE_ATOL = 1e-6


def _delta0(c_i: EligibleCandidate, c_j: EligibleCandidate) -> float:
    return c_i.y - c_j.y


def collect_pairs_for_task(
    cfg: ExperimentConfig, task_name_str: str, task_t: float, bank: list[EligibleCandidate],
    reference_checkpoint: Path, n_candidates: int, horizons: tuple[int, ...],
    rng: random.Random, work_dir: Path,
) -> list[dict]:
    """Every (i,j) pair, i<j, among n_candidates sampled from `bank`, with
    Delta_0 and Delta_h for every h in `horizons`, computed against the SAME
    M matched backgrounds for every h (background draw uses a horizon-
    independent seed, so h=1..50 are directly comparable realizations of the
    identical matched-pool comparison, not independently reshuffled ones).
    """
    if len(bank) < n_candidates + cfg.mi_num_backgrounds:
        return []
    candidates = rng.sample(bank, n_candidates)
    reserved = {c.seq for c in candidates}
    likelihoods = score_sequences(cfg, reference_checkpoint, task_t, [c.seq for c in bank])
    q = ReferenceAlignedDistribution(bank, likelihoods, cfg.mi_tau_q)

    # Background size = init_size - 1, exactly matching
    # mi_orpt/pair_construction.py::construct_pairs_for_task's own
    # `q.sample_background(cfg.init_size - 1, reserved, rng)` call -- there is
    # no separate "mi_background_size" field on ExperimentConfig; the real
    # pipeline derives it from init_size because a background + one candidate
    # must equal one evaluable init pool (m_pref = init_size).
    background_seed = rng.randrange(2**31 - 1)
    backgrounds, seeds = [], []
    for _ in range(cfg.mi_num_backgrounds):
        bg_rng = random.Random(background_seed + len(backgrounds))
        background = q.sample_background(cfg.init_size - 1, reserved, bg_rng)
        if background is None:
            return []
        backgrounds.append(background)
        seeds.append(background_seed + 1000 + len(backgrounds))

    utilities_by_h: dict[int, list[list[float]]] = {}
    for h in horizons:
        h_cfg = dataclasses.replace(cfg, mi_bo_steps=h)
        utilities_by_h[h] = run_candidates_one_step(
            h_cfg, task_t, backgrounds, candidates, work_dir / f"h{h}", seeds,
        )

    rows = []
    for i in range(len(candidates)):
        for j in range(i):
            delta0 = _delta0(candidates[i], candidates[j])
            row = {
                "task": task_name_str, "task_t": task_t,
                "candidate_i": candidates[i].seq, "candidate_j": candidates[j].seq,
                "y_i": candidates[i].y, "y_j": candidates[j].y, "delta0": delta0,
            }
            for h in horizons:
                diffs = [utilities_by_h[h][i][r] - utilities_by_h[h][j][r] for r in range(len(backgrounds))]
                mean = float(np.mean(diffs))
                se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("inf")
                row[f"delta_h{h}"] = mean
                row[f"se_h{h}"] = se
            rows.append(row)
    return rows


def collect_all(
    cfg: ExperimentConfig, task_indices: list[int], reference_milestone: int,
    n_candidates: int, horizons: tuple[int, ...], seed: int, work_dir: Path,
) -> list[dict]:
    reference_checkpoint = cfg.milestone_checkpoint_dir(reference_milestone)
    if not reference_checkpoint.exists():
        raise FileNotFoundError(f"reference checkpoint missing: {reference_checkpoint}")
    rng = random.Random(seed)
    all_rows = []
    for index in task_indices:
        name = task_name(index)
        task_t = cfg.train_task_values[index]
        bank = build_eligible_bank(cfg.trajectories_dir / f"{name}.csv")
        print(f"[ranking_reversal] {name} t={task_t:.3f} bank={len(bank)}", flush=True)
        rows = collect_pairs_for_task(
            cfg, name, task_t, bank, reference_checkpoint, n_candidates, horizons,
            random.Random(rng.randrange(2**31 - 1)), work_dir / name,
        )
        print(f"[ranking_reversal] {name}: {len(rows)} pairs", flush=True)
        all_rows.extend(rows)
    return all_rows


def reversal_and_agreement_stats(
    rows: list[dict], horizons: tuple[int, ...], *, reliability_filter: bool, z_min: float,
) -> dict:
    """reversal_rate(h) and agreement_Htrain(h) over `rows`, with a 95%
    Wilson-ish normal-approximation CI, plus raw counts -- per the doc's
    "also save: number of evaluated pairs, number retained after
    reliability filtering, reversal counts, agreement counts, confidence
    intervals."
    """
    def included(row: dict, h: int) -> bool:
        if abs(row["delta0"]) <= DELTA0_TIE_ATOL:
            return False
        if abs(row[f"delta_h{h}"]) <= DELTA0_TIE_ATOL:
            return False
        if reliability_filter:
            z = abs(row[f"delta_h{h}"]) / (row[f"se_h{h}"] + 1e-12)
            if z < z_min:
                return False
        return True

    out = {"n_pairs_total": len(rows), "reliability_filter": reliability_filter, "z_min": z_min, "by_h": {}}
    for h in horizons:
        kept = [r for r in rows if included(r, h)]
        n = len(kept)
        reversed_count = sum(1 for r in kept if np.sign(r["delta0"]) != np.sign(r[f"delta_h{h}"]))
        rate = reversed_count / n if n else float("nan")
        se = np.sqrt(rate * (1 - rate) / n) if n else float("nan")
        agree_train = None
        if h != TRAIN_HORIZON:
            kept_tr = [r for r in kept if abs(r[f"delta_h{TRAIN_HORIZON}"]) > DELTA0_TIE_ATOL]
            n_tr = len(kept_tr)
            agree_count = sum(
                1 for r in kept_tr if np.sign(r[f"delta_h{TRAIN_HORIZON}"]) == np.sign(r[f"delta_h{h}"])
            )
            agree_rate = agree_count / n_tr if n_tr else float("nan")
            agree_se = np.sqrt(agree_rate * (1 - agree_rate) / n_tr) if n_tr else float("nan")
            agree_train = {
                "n_pairs": n_tr, "agreement_count": agree_count, "agreement_rate": agree_rate,
                "ci95_low": agree_rate - 1.96 * agree_se, "ci95_high": agree_rate + 1.96 * agree_se,
            }
        out["by_h"][h] = {
            "n_pairs": n, "reversal_count": reversed_count, "reversal_rate": rate,
            "ci95_low": rate - 1.96 * se if n else float("nan"),
            "ci95_high": rate + 1.96 * se if n else float("nan"),
            "agreement_with_Htrain": agree_train,
        }
    return out


def save_raw_pairs(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))
    return path


def save_stats(stats: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))
    return path
