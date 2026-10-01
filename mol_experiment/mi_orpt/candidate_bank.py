"""Eligible candidate bank for mol's matched-intervention ORPT. Mirrors
peptide_experiment/mi_orpt/candidate_bank.py's shape (read-only reference, never
imported -- isolation contract) but with Tanimoto-to-seed feasibility instead of
edit-distance-to-reference, and reading mol_run_bo's own CSV format directly
rather than importing fine-tuning/peptides' make_dpo_train_data_csv.py /
make_train_data_csv.py.

EligibleCandidate/min_bank_size_needed are unchanged in shape from peptide's
(fully domain-generic -- `seq` here holds a canonical SMILES string, not an
amino-acid sequence, but nothing else cares which).
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "optimization" / "mol"))
from mol_fingerprint import tanimoto_similarity  # noqa: E402

PANDAS_DEFAULT_NA_TOKENS = {"", "NA", "N/A", "NaN", "nan", "null", "NULL"}


@dataclass(frozen=True)
class EligibleCandidate:
    seq: str
    y: float


def is_similar_enough(smiles: str, seed_smiles: str, tau_mol: float) -> bool:
    sim = tanimoto_similarity(smiles, seed_smiles)
    return sim is not None and sim >= tau_mol


def load_scored_sequences(
    input_csv: Path,
    seed_smiles: str,
    tau_mol: float,
    keep_infeasible: bool = False,
) -> list[tuple[str, float, bool]]:
    """Loads (smiles, score, is_feasible) triples from a mol_run_bo trajectory
    CSV (train_x, train_y columns). Mirrors
    make_dpo_train_data_csv.py::load_scored_sequences's contract exactly,
    against mol's own feasibility notion (Tanimoto-to-seed, not
    edit-distance-to-reference)."""
    scored_sequences = []
    rows_skipped = 0
    rows_infeasible = 0

    with input_csv.open(newline="") as f_in:
        reader = csv.DictReader(f_in)
        required_columns = {"train_x", "train_y"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Expected input CSV to contain columns {sorted(required_columns)}: {input_csv}")

        for row in reader:
            smiles = (row.get("train_x") or "").strip()
            if smiles in PANDAS_DEFAULT_NA_TOKENS:
                rows_skipped += 1
                continue
            try:
                score = float(row["train_y"])
            except ValueError:
                rows_skipped += 1
                continue
            if not math.isfinite(score):
                rows_skipped += 1
                continue

            is_feasible = is_similar_enough(smiles, seed_smiles, tau_mol)
            if not is_feasible:
                rows_infeasible += 1
                if not keep_infeasible:
                    continue

            scored_sequences.append((smiles, score, is_feasible))

    if len(scored_sequences) < 2:
        raise ValueError(f"Need at least 2 usable scored sequences in {input_csv}")

    kept_or_dropped = "kept for ranking" if keep_infeasible else "dropped"
    print(f"Input CSV: {input_csv}")
    print(f"  Loaded sequences: {len(scored_sequences)}")
    print(f"  Rows skipped (unparseable/NA/non-finite): {rows_skipped}")
    print(f"  Rows infeasible (Tanimoto < {tau_mol}): {rows_infeasible} -- {kept_or_dropped}")
    return scored_sequences


def build_eligible_bank(trajectory_csv: Path, seed_smiles: str, tau_mol: float) -> list[EligibleCandidate]:
    """Feasible, unique, finite-score candidates from one task's cumulative
    trajectory CSV. Dedup by canonical SMILES, first occurrence wins."""
    scored = load_scored_sequences(trajectory_csv, seed_smiles, tau_mol)
    seen = set()
    bank = []
    for smiles, score, _is_feasible in scored:
        if smiles in seen:
            continue
        seen.add(smiles)
        bank.append(EligibleCandidate(seq=smiles, y=score))
    return bank


def build_eligible_bank_from_init_scores(
    init_path: Path, scores_path: Path, seed_smiles: str, tau_mol: float
) -> list[EligibleCandidate]:
    """Same feasible/unique/first-occurrence-wins bank as build_eligible_bank(),
    sourced from a plain init.txt (one SMILES/line) + scores.csv (one
    score/line, matched by line index) pair -- the format
    mol_experiment/steps.py::mol_build_mutation_init() writes."""
    seqs = [line for line in init_path.read_text().splitlines() if line.strip()]
    scores = [float(line) for line in scores_path.read_text().splitlines() if line.strip()]
    if len(seqs) != len(scores):
        raise ValueError(f"{init_path} has {len(seqs)} sequences but {scores_path} has {len(scores)} scores")

    seen = set()
    bank = []
    for smiles, score in zip(seqs, scores):
        if smiles in seen:
            continue
        seen.add(smiles)
        if not is_similar_enough(smiles, seed_smiles, tau_mol):
            continue
        if not math.isfinite(score):
            continue
        bank.append(EligibleCandidate(seq=smiles, y=score))
    return bank


def min_bank_size_needed(m: int, num_reserved: int = 2) -> int:
    """Byte-identical logic to peptide_experiment/mi_orpt/candidate_bank.py's
    own min_bank_size_needed -- domain-generic combinatorics, no peptide content."""
    return (m - 1) + num_reserved
