"""Milestone 2, step 2: canonicalize ligand SMILES with RDKit, then aggregate
replicate measurements per (target_id, canonical_smiles) pair.

Per the spec's conservative default, censored IC50 records (qualifier '>' or '<')
are excluded here entirely -- not parsed as exact, not included in the aggregate.
They were already counted and reported in step1's report.json; this step only
processes the exact-measurement subset.

IC50 is converted to the same p-scale the frozen DeepPurpose oracle itself uses
(p = -log10(IC50_M), verified empirically in Milestone 1 against a real BindingDB
record), so seed potency and the oracle's own output are on a directly comparable
scale later. Replicates are aggregated by the median in this p-scale (per spec:
"median in log space" -- p-scale IS log space, log10 specifically).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_upstream" / "lolbo"))
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")  # invalid-molecule warnings are expected and counted below, not noise

IN_TSV = Path(__file__).resolve().parent / "intermediate" / "step1_filtered_rows.tsv"
OUT_DIR = Path(__file__).resolve().parent / "intermediate"
OUT_PAIRS = OUT_DIR / "step2_target_ligand_pairs.tsv"
OUT_REPORT = OUT_DIR / "step2_report.json"


def ic50_nM_to_p(ic50_nM: float) -> float:
    """p = -log10(IC50 in molar). Same convention DeepPurpose's MPNN_CNN_BindingDB_IC50
    was trained on (verified in Milestone 1: predicted 6.872 vs measured 6.752 p for a
    real BindingDB pair)."""
    return -math.log10(ic50_nM * 1e-9)


def main() -> None:
    n_censored_skipped = 0
    n_invalid_smiles = 0
    n_exact_rows_seen = 0
    n_nonpositive_ic50_skipped = 0

    # group[(target_id, canonical_smiles)] -> list of (p_value, target_name, sequence)
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    target_meta: dict[str, tuple[str, str]] = {}  # target_id -> (target_name, sequence)
    invalid_smiles_samples: list[str] = []

    with open(IN_TSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["ic50_qualifier"]:
                n_censored_skipped += 1
                continue
            n_exact_rows_seen += 1

            ic50_nM = float(row["ic50_nM"])
            if ic50_nM <= 0:
                # A handful of BindingDB rows carry a literal "0" IC50 -- physically
                # meaningless (undefined -log10) and almost certainly a curation
                # artifact, not a real measurement. Dropped explicitly and counted,
                # not clamped to some epsilon (F5/F6: no silent coercion).
                n_nonpositive_ic50_skipped += 1
                continue

            # BindingDB's "Ligand SMILES" field sometimes carries a trailing CXSMILES
            # extension block (e.g. " |THB:...|", " |TLB:...|", " |r|") for relative
            # stereochemistry. Chem.MolFromSmiles does not parse that block and
            # returns None for the whole string even though the base SMILES before
            # the first space is a perfectly valid molecule (verified directly: the
            # ~1800 "invalid" hits in an earlier run of this script were entirely
            # this pattern). Split it off before parsing rather than counting these
            # as chemically invalid.
            base_smiles = row["ligand_smiles"].split(" ", 1)[0]
            mol = Chem.MolFromSmiles(base_smiles)
            if mol is None:
                n_invalid_smiles += 1
                if len(invalid_smiles_samples) < 20:
                    invalid_smiles_samples.append(row["ligand_smiles"])
                continue
            canon = Chem.MolToSmiles(mol)

            p_val = ic50_nM_to_p(ic50_nM)
            groups[(row["target_id"], canon)].append(p_val)
            target_meta.setdefault(row["target_id"], (row["target_name"], row["sequence"]))

    # aggregate replicates: median in p-scale, retain count + spread
    import statistics

    with open(OUT_PAIRS, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["target_id", "target_name", "sequence", "canonical_smiles",
                          "p_median", "n_replicates", "p_stdev"])
        for (target_id, canon), p_values in groups.items():
            target_name, sequence = target_meta[target_id]
            p_median = statistics.median(p_values)
            p_stdev = statistics.pstdev(p_values) if len(p_values) > 1 else 0.0
            writer.writerow([target_id, target_name, sequence, canon,
                              f"{p_median:.6f}", len(p_values), f"{p_stdev:.6f}"])

    report = {
        "exact_rows_seen": n_exact_rows_seen,
        "nonpositive_ic50_rows_dropped": n_nonpositive_ic50_skipped,
        "censored_rows_skipped_here": n_censored_skipped,
        "invalid_smiles_rows_dropped": n_invalid_smiles,
        "invalid_smiles_samples": invalid_smiles_samples,
        "unique_target_ligand_pairs_after_canonicalization_and_aggregation": len(groups),
        "unique_targets": len(target_meta),
        "replicate_count_histogram": {},
    }
    from collections import Counter
    rep_counts = Counter(len(v) for v in groups.values())
    report["replicate_count_histogram"] = {str(k): v for k, v in sorted(rep_counts.items())[:20]}
    report["max_replicates_for_one_pair"] = max(len(v) for v in groups.values())

    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
