import argparse
import csv
import importlib.util
from pathlib import Path

from Levenshtein import distance as edit_distance


REFERENCE_SEQUENCE = "RRYYEQLEQASRKGNRGFRR"
TOP_N = 1000
SIMILARITY_THRESHOLD = 0.75

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


def is_similar_enough(sequence: str, reference_sequence: str, similarity_threshold: float) -> bool:
    length = len(reference_sequence)
    similarity = (length - edit_distance(sequence, reference_sequence)) / length
    return similarity >= similarity_threshold


def collect_top_sequences(
    input_csv: Path,
    top_n: int,
    reference_sequence: str,
    similarity_threshold: float,
) -> tuple[list[tuple[float, str]], int, int]:
    """Returns (top_n feasible sequences by score, rows_skipped as
    unparseable/NA, rows_skipped as infeasible under the similarity
    constraint). Feasibility filtering matches `APEXSimilarityConstraint` in
    `your_tasks/your_blackbox_constraints.py` -- without it, sequences that
    scored well on raw MIC but don't actually resemble the reference peptide
    (i.e. invalid solutions to the task) can end up as SFT training targets.
    """
    rows_skipped = 0
    rows_infeasible = 0
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

            if not is_similar_enough(sequence, reference_sequence, similarity_threshold):
                rows_infeasible += 1
                continue

            scored_sequences.append((score, sequence))

    top_sequences = sorted(scored_sequences, reverse=True)[:top_n]
    return top_sequences, rows_skipped, rows_infeasible


def make_train_data_csv(
    input_csvs: list[Path],
    output_csv: Path,
    reference_sequences: list[str],
    top_n: int,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if len(input_csvs) != len(reference_sequences):
        raise ValueError(
            f"Expected one reference sequence per input CSV, got "
            f"{len(input_csvs)} input CSVs and {len(reference_sequences)} references."
        )

    total_rows_written = 0

    with output_csv.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["sequence", "reference_sequence"])
        writer.writeheader()

        for input_csv, reference_sequence in zip(input_csvs, reference_sequences):
            top_sequences, rows_skipped, rows_infeasible = collect_top_sequences(
                input_csv, top_n, reference_sequence, similarity_threshold
            )

            for _, sequence in top_sequences:
                writer.writerow(
                    {
                        "sequence": sequence,
                        "reference_sequence": reference_sequence,
                    }
                )

            total_rows_written += len(top_sequences)
            print(f"Input CSV: {input_csv}")
            print(f"  Rows written: {len(top_sequences)}")
            print(f"  Rows skipped (unparseable/NA): {rows_skipped}")
            print(f"  Rows skipped (similarity < {similarity_threshold}): {rows_infeasible}")
            print(f"  Top score: {top_sequences[0][0] if top_sequences else 'n/a'}")
            print(
                f"  Bottom included score: "
                f"{top_sequences[-1][0] if top_sequences else 'n/a'}"
            )

    print(f"Wrote CSV: {output_csv}")
    print(f"Total rows written: {total_rows_written}")


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
        "--similarity-threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help="Minimum similarity to the reference sequence (matches the BO similarity "
        "constraint) a candidate must have to be eligible as an SFT training target.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.reference_index is not None:
        reference_sequences = [
            load_reference_sequence(reference_index)
            for reference_index in args.reference_index
        ]
    else:
        reference_sequences = [
            args.reference_sequence or REFERENCE_SEQUENCE
        ] * len(args.input_csv)

    make_train_data_csv(
        input_csvs=args.input_csv,
        output_csv=args.output_csv,
        reference_sequences=reference_sequences,
        top_n=args.top_n,
        similarity_threshold=args.similarity_threshold,
    )
