"""Milestone 2, step 3: tag every (target, canonical ligand) pair from step 2 with
whether its SELFIES encoding is fully covered by the LOL-BO SELFIES-VAE checkpoint's
fixed vocabulary (verified in Milestone 1: DEFAULT_SELFIES_VOCAB, 97 tokens).

This is a cheap, no-model check (selfies.encoder + split_selfies + set membership --
no VAE forward pass). Per the spec, only the ligand actually selected as a task's
seed needs to satisfy this ("A seed that fails this is dropped at task-construction
time, never at run time") -- but the eligible-candidate *pool* a target's seed is
chosen from is restricted to vocab-compatible ligands only, so that seed selection
never has to fall back ad hoc on a second-choice ligand at rule-application time.
Ligands are deduplicated by canonical SMILES before the check (many recur across
different targets) to avoid redundant work.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_upstream" / "lolbo"))
import selfies as sf  # noqa: E402
from lolbo.utils.mol_utils.selfies_vae.model_positional_unbounded import SELFIESDataset  # noqa: E402

IN_TSV = Path(__file__).resolve().parent / "intermediate" / "step2_target_ligand_pairs.tsv"
OUT_TSV = Path(__file__).resolve().parent / "intermediate" / "step3_pairs_with_vocab_flag.tsv"
OUT_REPORT = Path(__file__).resolve().parent / "intermediate" / "step3_report.json"


def main() -> None:
    dataobj = SELFIESDataset()
    vocab = dataobj.vocab2idx  # dict, membership check is O(1)

    # pass 1: unique canonical SMILES -> (ok: bool, selfies_str: str | None)
    unique_smiles: set[str] = set()
    with open(IN_TSV, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            unique_smiles.add(row["canonical_smiles"])

    result: dict[str, tuple[bool, str | None]] = {}
    n_selfies_encode_error = 0
    for smi in unique_smiles:
        try:
            selfies_str = sf.encoder(smi)
        except Exception:
            result[smi] = (False, None)
            n_selfies_encode_error += 1
            continue
        tokens = list(sf.split_selfies(selfies_str))
        ok = all(t in vocab for t in tokens)
        result[smi] = (ok, selfies_str if ok else None)

    n_unique = len(unique_smiles)
    n_unique_ok = sum(1 for ok, _ in result.values() if ok)

    # pass 2: join back onto every (target, ligand) row
    with open(IN_TSV, "r", newline="") as f_in, open(OUT_TSV, "w", newline="") as f_out:
        reader = csv.DictReader(f_in, delimiter="\t")
        fieldnames = reader.fieldnames + ["selfies_vocab_ok", "selfies_str"]
        writer = csv.DictWriter(f_out, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        n_rows = 0
        n_rows_ok = 0
        for row in reader:
            ok, selfies_str = result[row["canonical_smiles"]]
            row["selfies_vocab_ok"] = "1" if ok else "0"
            row["selfies_str"] = selfies_str or ""
            writer.writerow(row)
            n_rows += 1
            if ok:
                n_rows_ok += 1

    report = {
        "unique_canonical_ligands": n_unique,
        "unique_canonical_ligands_vocab_ok": n_unique_ok,
        "unique_canonical_ligands_vocab_ok_fraction": round(n_unique_ok / n_unique, 4),
        "selfies_encode_errors": n_selfies_encode_error,
        "total_target_ligand_rows": n_rows,
        "total_target_ligand_rows_vocab_ok": n_rows_ok,
    }
    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
