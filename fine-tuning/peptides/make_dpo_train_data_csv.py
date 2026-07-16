import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path


REFERENCE_SEQUENCE = "RRYYEQLEQASRKGNRGFRR"
DEFAULT_NUM_PAIRS = 1000
MAX_MUTATION_DISTANCE = 0.25
MIN_SCORE_GAP = 25.0

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
    "and amphipathicity. Respond with only the peptide sequence."
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


def edit_distance(first: str, second: str) -> int:
    if len(first) < len(second):
        first, second = second, first

    previous_row = list(range(len(second) + 1))
    for i, first_char in enumerate(first, start=1):
        current_row = [i]
        for j, second_char in enumerate(second, start=1):
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1
            substitution = previous_row[j - 1] + (first_char != second_char)
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row
    return previous_row[-1]


def mutation_distance(sequence: str, reference_sequence: str) -> float:
    return edit_distance(sequence, reference_sequence) / len(reference_sequence)


def load_scored_sequences(
    input_csv: Path,
    reference_sequence: str,
    max_mutation_distance: float,
) -> list[tuple[str, float, float]]:
    scored_sequences = []
    rows_skipped = 0
    invalid_for_pairing = 0

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

            distance = mutation_distance(sequence, reference_sequence)
            if len(sequence) != len(reference_sequence) or distance > max_mutation_distance:
                invalid_for_pairing += 1

            scored_sequences.append((sequence, score, distance))

    if len(scored_sequences) < 2:
        raise ValueError(
            f"Need at least 2 scored sequences within mutation distance "
            f"{max_mutation_distance} in {input_csv}"
        )

    print(f"Input CSV: {input_csv}")
    print(f"  Loaded sequences: {len(scored_sequences)}")
    print(f"  Rows skipped: {rows_skipped}")
    print(f"  Rows invalid for DPO pairing: {invalid_for_pairing}")
    return scored_sequences


def make_messages(reference_sequence: str, assistant_sequence: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": reference_sequence},
        {"role": "assistant", "content": assistant_sequence},
    ]


def sample_pairs(
    scored_sequences: list[tuple[str, float, float]],
    reference_sequence: str,
    num_pairs: int,
    rng: random.Random,
    min_score_gap: float,
    max_mutation_distance: float,
) -> list[dict]:
    pairs = []
    attempts = 0
    max_attempts = max(num_pairs * 200, 1000)
    valid_candidates = [
        scored_sequence
        for scored_sequence in scored_sequences
        if len(scored_sequence[0]) == len(reference_sequence)
        and scored_sequence[2] <= max_mutation_distance
    ]

    if len(valid_candidates) < 2:
        raise ValueError(
            f"Need at least 2 DPO candidates with same length as the reference "
            f"and mutation distance <= {max_mutation_distance}."
        )

    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        first, second = rng.sample(valid_candidates, 2)
        first_sequence, first_score, first_distance = first
        second_sequence, second_score, second_distance = second

        if first_score == second_score:
            continue

        if first_score > second_score:
            chosen_sequence, chosen_score, chosen_distance = first
            rejected_sequence, rejected_score, rejected_distance = second
        else:
            chosen_sequence, chosen_score, chosen_distance = second
            rejected_sequence, rejected_score, rejected_distance = first

        score_gap = chosen_score - rejected_score
        if score_gap < min_score_gap:
            continue

        pairs.append(
            {
                "reference_sequence": reference_sequence,
                "chosen_sequence": chosen_sequence,
                "rejected_sequence": rejected_sequence,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "score_gap": score_gap,
                "chosen_mutation_distance": chosen_distance,
                "rejected_mutation_distance": rejected_distance,
                "chosen": make_messages(reference_sequence, chosen_sequence),
                "rejected": make_messages(reference_sequence, rejected_sequence),
            }
        )

    if len(pairs) < num_pairs:
        raise ValueError(
            f"Could only create {len(pairs)} pairs out of requested {num_pairs}. "
            f"Too many tied scores or pairs with score gap below {min_score_gap} "
            f"may be present. Both chosen and rejected candidates are restricted "
            f"to same length as the reference and mutation distance <= "
            f"{max_mutation_distance}."
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
        "score_gap",
        "chosen_mutation_distance",
        "rejected_mutation_distance",
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
    parser.add_argument(
        "--max-mutation-distance",
        type=float,
        default=MAX_MUTATION_DISTANCE,
        help="Keep only sequences with edit_distance(sequence, reference) / len(reference) at or below this value.",
    )
    parser.add_argument(
        "--min-score-gap",
        type=float,
        default=MIN_SCORE_GAP,
        help="Keep only DPO pairs where chosen_score - rejected_score is at least this value.",
    )
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
        scored_sequences = load_scored_sequences(
            input_csv=input_csv,
            reference_sequence=reference_sequence,
            max_mutation_distance=args.max_mutation_distance,
        )
        all_pairs.extend(
            sample_pairs(
                scored_sequences=scored_sequences,
                reference_sequence=reference_sequence,
                num_pairs=args.pairs_per_input,
                rng=rng,
                min_score_gap=args.min_score_gap,
                max_mutation_distance=args.max_mutation_distance,
            )
        )

    write_csv(all_pairs, args.output_csv)
    write_jsonl(all_pairs, args.output_jsonl)
    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote JSONL: {args.output_jsonl}")
    print(f"Total DPO pairs: {len(all_pairs)}")
    print(f"Minimum score gap: {args.min_score_gap}")


if __name__ == "__main__":
    main()
