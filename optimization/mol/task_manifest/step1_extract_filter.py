"""Milestone 2, step 1: single streaming pass over the raw BindingDB_All.tsv dump.

Mirrors the peptide domain's apex_oracle/refseqs.py + apex_oracle/task_splits.py role
(building the task universe), but this file only owns the *first* pass: pull the
handful of columns we need, apply the row-level filters the spec requires, and
report counts after each filter. It does NOT aggregate/canonicalize/select seeds --
that is step2_aggregate_and_manifest.py, kept separate so this expensive full-file
scan runs exactly once.

No silent fallbacks (F1-F7 in opus_bindingdb_implementation_prompt.md):
  - an unrecognized IC50 qualifier character raises, it is not parsed as exact (F6).
  - every drop is counted and reported by an explicit reason code (F4).
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

RAW_TSV = Path(__file__).resolve().parents[3] / "data" / "mol" / "bindingdb" / "BindingDB_All.tsv"
OUT_DIR = Path(__file__).resolve().parent / "intermediate"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_ROWS = OUT_DIR / "step1_filtered_rows.tsv"
OUT_REPORT = OUT_DIR / "step1_report.json"

# Recognized BindingDB IC50 qualifier prefixes. Anything else in the numeric field
# that isn't a plain float and isn't one of these raises rather than being guessed at (F6).
KNOWN_QUALIFIERS = (">=", "<=", ">", "<", "~")


def parse_ic50_field(raw: str) -> tuple[float | None, str | None]:
    """Returns (value_nM, qualifier). qualifier is None for an exact measurement.
    Raises ValueError on a field that isn't empty, isn't a plain float, and doesn't
    start with a KNOWN_QUALIFIERS prefix -- an unrecognized format must not be
    silently coerced (F6)."""
    raw = raw.strip()
    if not raw:
        return None, None
    for q in KNOWN_QUALIFIERS:
        if raw.startswith(q):
            rest = raw[len(q):].strip()
            return float(rest), q
    # plain numeric (BindingDB pads some with a leading space, already stripped above)
    return float(raw), None


def main() -> None:
    if not RAW_TSV.exists():
        raise FileNotFoundError(f"expected raw BindingDB dump at {RAW_TSV} (Milestone 1 artifact)")

    counts: dict[str, int] = {}
    qualifier_counts: Counter[str] = Counter()
    unrecognized_qualifier_samples: list[str] = []
    protein_len_samples: list[int] = []

    with open(RAW_TSV, "r", encoding="utf-8", errors="replace") as f_in, \
         open(OUT_ROWS, "w", encoding="utf-8", newline="") as f_out:
        reader = csv.reader(f_in, delimiter="\t")
        header = next(reader)

        def col(name: str) -> int:
            return header.index(name)

        idx = {
            "smiles": col("Ligand SMILES"),
            "ic50": col("IC50 (nM)"),
            "seq1": col("BindingDB Target Chain Sequence 1"),
            "nchains": col("Number of Protein Chains in Target (>1 implies a multichain complex)"),
            "uniprot_sp": col("UniProt (SwissProt) Primary ID of Target Chain 1"),
            "uniprot_trembl": col("UniProt (TrEMBL) Primary ID of Target Chain 1"),
            "target_name": col("Target Name"),
        }

        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(["target_id", "target_id_source", "target_name", "sequence",
                          "ligand_smiles", "ic50_nM", "ic50_qualifier"])

        n_raw = 0
        n_single_chain = 0
        n_has_smiles = 0
        n_has_sequence = 0
        n_has_target_id = 0
        n_has_ic50_field = 0
        n_ic50_parsed = 0
        n_ic50_censored = 0
        n_ic50_exact = 0
        n_written = 0

        for row in reader:
            n_raw += 1
            if len(row) <= max(idx.values()):
                continue  # malformed/short row; excluded, not coerced

            if row[idx["nchains"]].strip() != "1":
                continue
            n_single_chain += 1

            smiles = row[idx["smiles"]].strip()
            if not smiles:
                continue
            n_has_smiles += 1

            sequence = row[idx["seq1"]].strip()
            if not sequence:
                continue
            n_has_sequence += 1
            protein_len_samples.append(len(sequence))

            sp_id = row[idx["uniprot_sp"]].strip()
            trembl_id = row[idx["uniprot_trembl"]].strip()
            if sp_id:
                target_id, target_id_source = sp_id, "swissprot"
            elif trembl_id:
                target_id, target_id_source = trembl_id, "trembl"
            else:
                continue
            n_has_target_id += 1

            ic50_raw = row[idx["ic50"]]
            if not ic50_raw.strip():
                continue
            n_has_ic50_field += 1

            try:
                ic50_val, qualifier = parse_ic50_field(ic50_raw)
            except ValueError:
                unrecognized_qualifier_samples.append(ic50_raw)
                continue
            if ic50_val is None:
                continue
            n_ic50_parsed += 1
            qualifier_counts[qualifier or "<exact>"] += 1
            if qualifier is None:
                n_ic50_exact += 1
            else:
                n_ic50_censored += 1

            writer.writerow([
                target_id, target_id_source, row[idx["target_name"]].strip(),
                sequence, smiles, ic50_val, qualifier or "",
            ])
            n_written += 1

    counts = {
        "raw_rows": n_raw,
        "after_single_chain_filter": n_single_chain,
        "after_nonempty_smiles_filter": n_has_smiles,
        "after_nonempty_sequence_filter": n_has_sequence,
        "after_stable_target_id_filter": n_has_target_id,
        "after_nonempty_ic50_field_filter": n_has_ic50_field,
        "after_ic50_parse": n_ic50_parsed,
        "ic50_exact_count": n_ic50_exact,
        "ic50_censored_count": n_ic50_censored,
        "unrecognized_ic50_field_count": len(unrecognized_qualifier_samples),
        "rows_written": n_written,
        "qualifier_frequency": dict(qualifier_counts),
        "unrecognized_ic50_field_samples": unrecognized_qualifier_samples[:20],
        "protein_length_min": min(protein_len_samples) if protein_len_samples else None,
        "protein_length_max": max(protein_len_samples) if protein_len_samples else None,
    }

    with open(OUT_REPORT, "w") as f:
        json.dump(counts, f, indent=2)

    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
