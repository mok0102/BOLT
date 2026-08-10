"""Eligible candidate bank for matched-intervention ORPT
(imp_plan/06_orpt_matched_intervention_plan.md, updated for the one-step
actual-BO evaluator -- paper/appendix.tex's app:candidate-banks: "No
separate future-acquisition candidate bank is required because the
one-step BO operator optimizes acquisition over the deployment search
space." One eligible set serves both shared-background sampling and
intervention candidates; there is no pool/eval role split.

Reuses fine-tuning/peptides/make_dpo_train_data_csv.py's own
load_scored_sequences() (same feasibility notion the rest of ORPT already
uses) rather than re-implementing similarity filtering.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

FINE_TUNING_DIR = Path(__file__).resolve().parents[2] / "fine-tuning" / "peptides"
if str(FINE_TUNING_DIR) not in sys.path:
    sys.path.insert(0, str(FINE_TUNING_DIR))

from make_dpo_train_data_csv import load_scored_sequences  # noqa: E402


@dataclass(frozen=True)
class EligibleCandidate:
    seq: str
    y: float


def build_eligible_bank(trajectory_csv: Path, reference_sequence: str, similarity_threshold: float) -> list[EligibleCandidate]:
    """Feasible, unique, finite-score candidates from one task's cumulative
    trajectory CSV. Dedup by sequence, first occurrence wins (real
    trajectory CSVs run heavily duplicated)."""
    scored = load_scored_sequences(trajectory_csv, reference_sequence, similarity_threshold)
    seen = set()
    bank = []
    for seq, score, _is_feasible in scored:
        if seq in seen:
            continue
        seen.add(seq)
        bank.append(EligibleCandidate(seq=seq, y=score))
    return bank


def min_bank_size_needed(m: int, num_reserved: int = 2) -> int:
    """A task needs at least (m-1)+num_reserved eligible candidates: the
    m-1 shared-background candidates plus num_reserved intervention
    candidates, all drawn from and excluded out of the same eligible set
    (appendix.tex app:candidate-banks: "Tasks with fewer than m+1 eligible
    unique candidates ... are omitted from pair construction").
    num_reserved=2 (default) matches that original m+1 rule -- the two
    candidates of a single matched pair. mi_orpt/pair_construction.py's
    cache-and-reuse design instead reserves up to mi_max_candidates_per_task
    candidates upfront (so every reserved candidate can be safely excluded
    from every shared background at once), passing that as num_reserved."""
    return (m - 1) + num_reserved
