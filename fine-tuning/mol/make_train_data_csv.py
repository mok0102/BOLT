"""Mol analogue of fine-tuning/peptides/make_train_data_csv.py: builds the
cumulative BOLT-SFT training CSV from tasks [0, milestone)'s trajectory
data (read-only reference to that script, never imported -- isolation
contract). Differs from it in:

- Tanimoto-to-seed feasibility (optimization/mol/mol_fingerprint.py's
  tanimoto_similarity) instead of Levenshtein-edit-distance-to-reference.
- the assistant completion must be a SELFIES string (optimization/mol/
  mol_prompt.py's system prompt requires "Output only the candidate SELFIES
  string"), so the winning SMILES from each task's trajectory CSV is
  re-encoded to SELFIES and vocab-checked (same selfies.encoder +
  split_selfies + vocab-membership pattern as optimization/mol/
  task_manifest/step3_selfies_vocab_check.py) before being written out; a
  candidate whose winning SMILES fails this check is dropped (the next-best
  feasible candidate takes its place) rather than silently emitting an SFT
  target the LLM's own required output grammar could never reproduce.
- per-task context is (target_sequence, seed_selfies), not a single
  reference string -- both are carried as their own CSV columns so
  generate_openai_ft_data.py can call mol_prompt.py's make_messages exactly
  as sampling and reference-model scoring do, instead of duplicating prompt
  text here.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

MOL_DIR = Path(__file__).resolve().parents[2] / "optimization" / "mol"
sys.path.insert(0, str(MOL_DIR))

from mol_fingerprint import tanimoto_similarity  # noqa: E402
from mol_lolbo.utils.mol_utils.selfies_vae.model_positional_unbounded import SELFIESDataset  # noqa: E402
from mol_tasks import MolTask, get_task  # noqa: E402

import selfies as sf  # noqa: E402

TOP_N = 1000
TAU_MOL = 0.4

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "train_data" / "train_data.csv"

# pandas.read_csv treats these strings as missing values by default -- kept
# in sync with candidate_bank.py's own PANDAS_DEFAULT_NA_TOKENS.
PANDAS_DEFAULT_NA_TOKENS = {"", "NA", "N/A", "NaN", "nan", "null", "NULL"}

_VOCAB = None


def is_similar_enough(smiles: str, seed_smiles: str, tau_mol: float) -> bool:
    sim = tanimoto_similarity(smiles, seed_smiles)
    return sim is not None and sim >= tau_mol


def smiles_to_selfies_vocab_checked(smiles: str) -> str | None:
    """Returns the SELFIES encoding of `smiles` if every token is covered by
    the LOL-BO SELFIES-VAE checkpoint's fixed vocabulary, else None."""
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = SELFIESDataset().vocab2idx
    try:
        selfies_str = sf.encoder(smiles)
    except Exception:
        return None
    if not all(t in _VOCAB for t in sf.split_selfies(selfies_str)):
        return None
    return selfies_str


def collect_top_sequences(
    input_csv: Path,
    top_n: int,
    seed_smiles: str,
    tau_mol: float,
) -> tuple[list[tuple[float, str]], int, int, int]:
    """Returns (top_n feasible (score, selfies) pairs, rows_skipped as
    unparseable/NA/non-finite, rows_skipped as infeasible under the Tanimoto
    constraint, rows_skipped as failing the SELFIES vocab check)."""
    rows_skipped = 0
    rows_infeasible = 0
    scored_sequences = []

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
            if not is_similar_enough(smiles, seed_smiles, tau_mol):
                rows_infeasible += 1
                continue
            scored_sequences.append((score, smiles))

    scored_sequences.sort(reverse=True)
    top_selfies: list[tuple[float, str]] = []
    rows_selfies_reject = 0
    for score, smiles in scored_sequences:
        if len(top_selfies) >= top_n:
            break
        selfies_str = smiles_to_selfies_vocab_checked(smiles)
        if selfies_str is None:
            rows_selfies_reject += 1
            continue
        top_selfies.append((score, selfies_str))

    return top_selfies, rows_skipped, rows_infeasible, rows_selfies_reject


def make_train_data_csv(
    input_csvs: list[Path],
    output_csv: Path,
    tasks: list[MolTask],
    top_n: int,
    tau_mol: float = TAU_MOL,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if len(input_csvs) != len(tasks):
        raise ValueError(f"Expected one task per input CSV, got {len(input_csvs)} input CSVs and {len(tasks)} tasks.")

    total_rows_written = 0

    with output_csv.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["candidate_selfies", "target_sequence", "seed_selfies"])
        writer.writeheader()

        for input_csv, task in zip(input_csvs, tasks):
            top_selfies, rows_skipped, rows_infeasible, rows_selfies_reject = collect_top_sequences(
                input_csv, top_n, task.seed_smiles, tau_mol
            )

            for _, selfies_str in top_selfies:
                writer.writerow(
                    {
                        "candidate_selfies": selfies_str,
                        "target_sequence": task.sequence,
                        "seed_selfies": task.seed_selfies,
                    }
                )

            total_rows_written += len(top_selfies)
            print(f"Input CSV: {input_csv} (task {task.task_idx})")
            print(f"  Rows written: {len(top_selfies)}")
            print(f"  Rows skipped (unparseable/NA/non-finite): {rows_skipped}")
            print(f"  Rows skipped (Tanimoto < {tau_mol}): {rows_infeasible}")
            print(f"  Rows rejected (SELFIES vocab check failed): {rows_selfies_reject}")
            print(f"  Top score: {top_selfies[0][0] if top_selfies else 'n/a'}")
            print(f"  Bottom included score: {top_selfies[-1][0] if top_selfies else 'n/a'}")

    print(f"Wrote CSV: {output_csv}")
    print(f"Total rows written: {total_rows_written}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train_data.csv for generate_openai_ft_data.py from mol BindingDB optimization data."
    )
    parser.add_argument("--input-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--task-index", type=int, nargs="+", required=True, help="Index into optimization/mol/mol_tasks.py's manifest.")
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument(
        "--tau-mol",
        type=float,
        default=TAU_MOL,
        help="Minimum Tanimoto similarity to the seed ligand (matches the BO feasibility "
        "constraint) a candidate must have to be eligible as an SFT training target.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tasks = [get_task(i) for i in args.task_index]
    make_train_data_csv(
        input_csvs=args.input_csv,
        output_csv=args.output_csv,
        tasks=tasks,
        top_n=args.top_n,
        tau_mol=args.tau_mol,
    )
