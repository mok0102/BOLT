import argparse
import csv
import importlib.util
import random
import re
from pathlib import Path


REFERENCE_SEQUENCE = "RRYYEQLEQASRKGNRGFRR"
TOP_N = 1000
MAX_MUTATION_DISTANCE = 0.25

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = (
    SCRIPT_DIR
    / "../../optimization/peptides/lolbo_scripts/optimization_all_collected_data/"
    / "BOLT-apex_no-wandb-tracking_all-data-collected.csv"
).resolve()
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "train_data" / "train_data.csv"
REFSEQS_PATH = (
    SCRIPT_DIR / "../../optimization/peptides/apex_oracle/refseqs.py"
).resolve()

# pandas.read_csv treats these strings as missing values by default.
# Skip them so generate_openai_ft_data.py can read the output without
# turning peptide strings such as "NA" into NaN.
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


def collect_top_sequences(
    input_csv: Path,
    reference_sequence: str,
    top_n: int,
    max_mutation_distance: float,
) -> tuple[list[tuple[float, str, float]], int, int]:
    rows_skipped = 0
    distance_skipped = 0
    scored_sequences = []

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
            if distance > max_mutation_distance:
                distance_skipped += 1
                continue

            scored_sequences.append((score, sequence, distance))

    top_sequences = sorted(scored_sequences, reverse=True)[:top_n]
    return top_sequences, rows_skipped, distance_skipped


def make_train_data_csv(
    input_csvs: list[Path],
    output_csv: Path,
    reference_sequences: list[str],
    reference_indices: list[int | None],
    top_n: int,
    max_mutation_distance: float,
    train_n: int | None,
    seed: int,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if len(input_csvs) != len(reference_sequences):
        raise ValueError(
            f"Expected one reference sequence per input CSV, got "
            f"{len(input_csvs)} input CSVs and {len(reference_sequences)} references."
        )

    output_rows = []

    for input_csv, reference_sequence, reference_index in zip(
        input_csvs, reference_sequences, reference_indices
    ):
        top_sequences, rows_skipped, distance_skipped = collect_top_sequences(
            input_csv=input_csv,
            reference_sequence=reference_sequence,
            top_n=top_n,
            max_mutation_distance=max_mutation_distance,
        )

        for score, sequence, distance in top_sequences:
            output_rows.append(
                {
                    "reference_index": reference_index,
                    "sequence": sequence,
                    "reference_sequence": reference_sequence,
                    "y_value": score,
                    "mutation_distance": distance,
                }
            )

        print(f"Input CSV: {input_csv}")
        print(f"  Reference index: {reference_index}")
        print(f"  Reference sequence: {reference_sequence}")
        print(f"  Candidate rows collected: {len(top_sequences)}")
        print(f"  Rows skipped: {rows_skipped}")
        print(f"  Rows skipped by mutation distance: {distance_skipped}")
        print(f"  Top score: {top_sequences[0][0] if top_sequences else 'n/a'}")
        print(
            f"  Bottom included score: "
            f"{top_sequences[-1][0] if top_sequences else 'n/a'}"
        )

    total_candidate_rows = len(output_rows)
    if train_n is not None:
        if train_n < 1:
            raise ValueError("--train-n must be at least 1 when provided.")
        if train_n > total_candidate_rows:
            raise ValueError(
                f"--train-n={train_n} is larger than the candidate pool size "
                f"{total_candidate_rows}."
            )
        rng = random.Random(seed)
        output_rows = rng.sample(output_rows, train_n)
        output_rows.sort(
            key=lambda row: (row["reference_index"] is None, row["reference_index"], -row["y_value"])
        )

    with output_csv.open("w", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=[
                "reference_index",
                "sequence",
                "reference_sequence",
                "y_value",
                "mutation_distance",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote CSV: {output_csv}")
    print(f"Total candidate rows: {total_candidate_rows}")
    print(f"Total rows written: {len(output_rows)}")


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


def infer_reference_index_from_path(input_csv: Path) -> int | None:
    match = re.search(r"(?:seed|task)_(\d+)", input_csv.name)
    if match:
        return int(match.group(1))
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train_data.csv for generate_openai_ft_data.py from BOLT peptide optimization data."
    )
    parser.add_argument("--input-csv", type=Path, nargs="+", default=[DEFAULT_INPUT_CSV])
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--reference-sequence", "--reference_sequence", default=None)
    parser.add_argument(
        "--reference-index",
        type=int,
        nargs="+",
        default=None,
        help="Index into apex_oracle/refseqs.py. Overrides --reference-sequence.",
    )
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument(
        "--train-n",
        type=int,
        default=None,
        help="Randomly sample this many rows from the collected top-n candidate pool.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --train-n is provided.",
    )
    parser.add_argument(
        "--max-mutation-distance",
        type=float,
        default=MAX_MUTATION_DISTANCE,
        help="Keep only sequences with edit_distance(sequence, reference) / len(reference) at or below this value.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.reference_index is not None:
        if len(args.reference_index) != len(args.input_csv):
            raise ValueError(
                f"Expected one --reference-index per --input-csv, got "
                f"{len(args.reference_index)} indices and {len(args.input_csv)} CSVs."
            )
        reference_indices = args.reference_index
        reference_sequences = [
            load_reference_sequence(reference_index)
            for reference_index in args.reference_index
        ]
    else:
        inferred_reference_indices = [
            infer_reference_index_from_path(input_csv) for input_csv in args.input_csv
        ]
        if args.reference_sequence is None and all(
            reference_index is not None for reference_index in inferred_reference_indices
        ):
            reference_indices = inferred_reference_indices
            reference_sequences = [
                load_reference_sequence(reference_index)
                for reference_index in reference_indices
                if reference_index is not None
            ]
        else:
            reference_indices = [None] * len(args.input_csv)
            reference_sequences = [args.reference_sequence or REFERENCE_SEQUENCE] * len(
                args.input_csv
            )

    make_train_data_csv(
        input_csvs=args.input_csv,
        output_csv=args.output_csv,
        reference_sequences=reference_sequences,
        reference_indices=reference_indices,
        top_n=args.top_n,
        max_mutation_distance=args.max_mutation_distance,
        train_n=args.train_n,
        seed=args.seed,
    )
