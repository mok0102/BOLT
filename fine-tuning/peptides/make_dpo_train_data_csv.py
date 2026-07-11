import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path


REFERENCE_SEQUENCE = "RRYYEQLEQASRKGNRGFRR"
DEFAULT_NUM_PAIRS = 1000

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = (
    SCRIPT_DIR
    / "../../optimization/peptides/lolbo_scripts/optimization_all_collected_data/"
    / "BOLT-apex_no-wandb-tracking_all-data-collected.csv"
).resolve()
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "train_data" / "dpo_train_data.csv"
DEFAULT_OUTPUT_JSONL = SCRIPT_DIR / "train_data" / "dpo_train_data.jsonl"
REFSEQS_PATH = (
    SCRIPT_DIR / "../../optimization/peptides/apex_oracle/refseqs.py"
).resolve()

SYSTEM_PROMPT = (
    "You are a specialized assistant that modifies peptide sequences to enhance "
    "antimicrobial activity. Make up to 25% sequence modifications based on known "
    "antimicrobial peptide properties such as: positive charge, hydrophobicity, "
    "and amphipathicity."
)

PANDAS_DEFAULT_NA_TOKENS = {
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
}


def load_reference_sequence(reference_index: int) -> str:
    spec = importlib.util.spec_from_file_location("apex_refseqs", REFSEQS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load reference sequences from {REFSEQS_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference_sequences = list(module.REFERENCE_SEQUENCE)

    if reference_index < 0 or reference_index >= len(reference_sequences):
        raise ValueError(
            f"reference_index={reference_index} is out of range for "
            f"{len(reference_sequences)} reference sequences."
        )
    return reference_sequences[reference_index]


def load_scored_sequences(input_csv: Path) -> list[tuple[str, float]]:
    scored_sequences = []
    rows_skipped = 0

    with input_csv.open(newline="") as f_in:
        reader = csv.DictReader(f_in)
        required_columns = {"train_x", "train_y"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Expected input CSV to contain columns {sorted(required_columns)}: {input_csv}"
            )

        for row in reader:
            sequence = (row.get("train_x") or "").strip()
            if sequence in PANDAS_DEFAULT_NA_TOKENS:
                rows_skipped += 1
                continue

            try:
                score = float(row["train_y"])
            except ValueError:
                rows_skipped += 1
                continue

            scored_sequences.append((sequence, score))

    if len(scored_sequences) < 2:
        raise ValueError(f"Need at least 2 scored sequences in {input_csv}")

    print(f"Input CSV: {input_csv}")
    print(f"  Loaded sequences: {len(scored_sequences)}")
    print(f"  Rows skipped: {rows_skipped}")
    return scored_sequences


def make_messages(reference_sequence: str, assistant_sequence: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": reference_sequence},
        {"role": "assistant", "content": assistant_sequence},
    ]


def sample_pairs(
    scored_sequences: list[tuple[str, float]],
    reference_sequence: str,
    num_pairs: int,
    rng: random.Random,
) -> list[dict]:
    pairs = []
    attempts = 0
    max_attempts = max(num_pairs * 20, 100)

    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        first, second = rng.sample(scored_sequences, 2)
        first_sequence, first_score = first
        second_sequence, second_score = second

        if first_score == second_score:
            continue

        if first_score > second_score:
            chosen_sequence, chosen_score = first_sequence, first_score
            rejected_sequence, rejected_score = second_sequence, second_score
        else:
            chosen_sequence, chosen_score = second_sequence, second_score
            rejected_sequence, rejected_score = first_sequence, first_score

        pairs.append(
            {
                "reference_sequence": reference_sequence,
                "chosen_sequence": chosen_sequence,
                "rejected_sequence": rejected_sequence,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "chosen": make_messages(reference_sequence, chosen_sequence),
                "rejected": make_messages(reference_sequence, rejected_sequence),
            }
        )

    if len(pairs) < num_pairs:
        raise ValueError(
            f"Could only create {len(pairs)} pairs out of requested {num_pairs}. "
            "Too many tied scores may be present."
        )
    return pairs


def write_csv(pairs: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reference_sequence",
        "chosen_sequence",
        "rejected_sequence",
        "chosen_score",
        "rejected_score",
    ]
    with output_csv.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow({fieldname: pair[fieldname] for fieldname in fieldnames})


def write_jsonl(pairs: list[dict], output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as f_out:
        for pair in pairs:
            f_out.write(
                json.dumps(
                    {
                        "chosen": pair["chosen"],
                        "rejected": pair["rejected"],
                    }
                )
                + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create DPO preference data from BOLT peptide optimization CSVs."
    )
    parser.add_argument("--input-csv", type=Path, nargs="+", default=[DEFAULT_INPUT_CSV])
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--reference-sequence", "--reference_sequence", default=None)
    parser.add_argument(
        "--reference-index",
        type=int,
        nargs="+",
        default=None,
        help="Index into apex_oracle/refseqs.py. Overrides --reference-sequence.",
    )
    parser.add_argument(
        "--pairs-per-input",
        type=int,
        default=DEFAULT_NUM_PAIRS,
        help="Random DPO pairs to sample from each input CSV.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.reference_index is not None:
        reference_sequences = [
            load_reference_sequence(reference_index)
            for reference_index in args.reference_index
        ]
    else:
        reference_sequences = [
            args.reference_sequence or REFERENCE_SEQUENCE
        ] * len(args.input_csv)

    if len(args.input_csv) != len(reference_sequences):
        raise ValueError(
            f"Expected one reference per input CSV, got {len(args.input_csv)} "
            f"input CSVs and {len(reference_sequences)} references."
        )

    all_pairs = []
    for input_csv, reference_sequence in zip(args.input_csv, reference_sequences):
        scored_sequences = load_scored_sequences(input_csv)
        all_pairs.extend(
            sample_pairs(
                scored_sequences=scored_sequences,
                reference_sequence=reference_sequence,
                num_pairs=args.pairs_per_input,
                rng=rng,
            )
        )

    write_csv(all_pairs, args.output_csv)
    write_jsonl(all_pairs, args.output_jsonl)
    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote JSONL: {args.output_jsonl}")
    print(f"Total DPO pairs: {len(all_pairs)}")


if __name__ == "__main__":
    main()
