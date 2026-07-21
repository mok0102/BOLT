import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path

from make_train_data_csv import is_similar_enough

REFERENCE_SEQUENCE = "RRYYEQLEQASRKGNRGFRR"
DEFAULT_NUM_PAIRS = 1000
SIMILARITY_THRESHOLD = 0.75

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


PAIRING_MODES = ("feasible_only", "lexicographic")


def load_scored_sequences(
    input_csv: Path,
    reference_sequence: str,
    similarity_threshold: float,
    pairing_mode: str = "feasible_only",
) -> list[tuple[str, float, bool]]:
    """Loads (sequence, score, is_feasible) triples.

    pairing_mode="feasible_only" (default, original behavior): rows that
    don't satisfy the similarity constraint against `reference_sequence` are
    dropped entirely, so every returned triple has is_feasible=True --
    preference pairs are never built from candidates that scored well but
    don't actually resemble the reference peptide. Matches the feasibility
    notion `make_train_data_csv.py`'s `collect_top_sequences()` already
    applies to the SFT dataset.

    pairing_mode="lexicographic": infeasible rows are kept (tagged
    is_feasible=False) instead of dropped, so sample_pairs() can rank
    feasibility ahead of the objective score -- see its docstring for why
    (naive objective-only ranking among feasible-only pairs still lets the
    trained policy propose infeasible sequences at generation time; see
    /root/.claude/plans/orpt-beta-0-25-parsed-turing.md and
    experiments/constraint_violation/).
    """
    assert pairing_mode in PAIRING_MODES, pairing_mode
    scored_sequences = []
    rows_skipped = 0
    rows_infeasible = 0

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

            is_feasible = is_similar_enough(sequence, reference_sequence, similarity_threshold)
            if not is_feasible:
                rows_infeasible += 1
                if pairing_mode == "feasible_only":
                    continue

            scored_sequences.append((sequence, score, is_feasible))

    if len(scored_sequences) < 2:
        raise ValueError(f"Need at least 2 usable scored sequences in {input_csv}")

    kept_or_dropped = "kept for lexicographic ranking" if pairing_mode == "lexicographic" else "dropped"
    print(f"Input CSV: {input_csv}")
    print(f"  Loaded sequences: {len(scored_sequences)}")
    print(f"  Rows skipped (unparseable/NA): {rows_skipped}")
    print(f"  Rows infeasible (similarity < {similarity_threshold}): {rows_infeasible} -- {kept_or_dropped}")
    return scored_sequences


def make_messages(reference_sequence: str, assistant_sequence: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": reference_sequence},
        {"role": "assistant", "content": assistant_sequence},
    ]


def _pick_chosen_rejected(
    first: tuple[str, float, bool],
    second: tuple[str, float, bool],
    pairing_mode: str,
) -> tuple[str, float, str, float] | None:
    """Returns (chosen_seq, chosen_score, rejected_seq, rejected_score) for
    this sampled candidate pair, or None if it carries no training signal.

    pairing_mode="feasible_only": both candidates are already guaranteed
    feasible by load_scored_sequences(), so this is just the original
    objective-score ranking (higher score wins; ties carry no signal).

    pairing_mode="lexicographic": feasibility is ranked *ahead of* the
    objective score -- exactly one candidate being feasible wins that pair
    regardless of score (this is the fix for naive DPO's constraint
    violations: it directly teaches "infeasible loses" independent of how
    good the objective looks). Both infeasible carries no useful signal
    (skipped). Both feasible falls through to the same score-based rule as
    feasible_only.
    """
    first_sequence, first_score, first_feasible = first
    second_sequence, second_score, second_feasible = second

    if pairing_mode == "lexicographic" and first_feasible != second_feasible:
        if first_feasible:
            return first_sequence, first_score, second_sequence, second_score
        return second_sequence, second_score, first_sequence, first_score

    if pairing_mode == "lexicographic" and not first_feasible and not second_feasible:
        return None

    if first_score == second_score:
        return None
    if first_score > second_score:
        return first_sequence, first_score, second_sequence, second_score
    return second_sequence, second_score, first_sequence, first_score


def _build_pair(reference_sequence: str, result: tuple[str, float, str, float]) -> dict:
    chosen_sequence, chosen_score, rejected_sequence, rejected_score = result
    return {
        "reference_sequence": reference_sequence,
        "chosen_sequence": chosen_sequence,
        "rejected_sequence": rejected_sequence,
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "chosen": make_messages(reference_sequence, chosen_sequence),
        "rejected": make_messages(reference_sequence, rejected_sequence),
    }


def sample_pairs(
    scored_sequences: list[tuple[str, float, bool]],
    reference_sequence: str,
    num_pairs: int,
    rng: random.Random,
    pairing_mode: str = "feasible_only",
) -> list[dict]:
    max_attempts = max(num_pairs * 50, 200)

    if pairing_mode == "lexicographic":
        feasible = [s for s in scored_sequences if s[2]]
        infeasible = [s for s in scored_sequences if not s[2]]
        if not feasible:
            raise ValueError(
                "No feasible sequences available for lexicographic pairing "
                f"(0 of {len(scored_sequences)} candidates satisfy the similarity "
                "constraint) -- every pair would be both-infeasible with no signal."
            )

        # Draw pairs stratified by feasibility instead of blindly sampling 2
        # rows from the whole pool: a mixed (feasible, infeasible) draw is
        # always a valid pair, so this can't stall the way naive rejection
        # sampling does when the feasible fraction is extreme (e.g. 50 of
        # 10050 candidates, ~0.5%, seen on real trajectory data at low
        # milestones). Mixed vs both-feasible draws are weighted by how many
        # of each actually exist (nf*ninf vs nf*(nf-1)), which is the same
        # ratio a uniform draw over the whole pool would produce among its
        # valid (non-both-infeasible) outcomes -- so this changes only the
        # sampling *mechanism*, not the resulting pair-type distribution.
        n_mixed = len(feasible) * len(infeasible)
        n_both_feasible = len(feasible) * (len(feasible) - 1)
        total = n_mixed + n_both_feasible
        mixed_probability = n_mixed / total if total else 0.0

        pairs = []
        attempts = 0
        while len(pairs) < num_pairs and attempts < max_attempts:
            attempts += 1
            if infeasible and rng.random() < mixed_probability:
                first, second = rng.choice(feasible), rng.choice(infeasible)
            elif len(feasible) >= 2:
                first, second = rng.sample(feasible, 2)
            else:
                continue
            result = _pick_chosen_rejected(first, second, pairing_mode)
            if result is None:
                continue
            pairs.append(_build_pair(reference_sequence, result))

        if len(pairs) < num_pairs:
            raise ValueError(
                f"Could only create {len(pairs)} pairs out of requested {num_pairs}. "
                "Too many tied scores among feasible candidates may be present."
            )
        return pairs

    pairs = []
    attempts = 0
    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        first, second = rng.sample(scored_sequences, 2)
        result = _pick_chosen_rejected(first, second, pairing_mode)
        if result is None:
            continue
        pairs.append(_build_pair(reference_sequence, result))

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
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help="Minimum similarity to the reference sequence (matches the BO similarity "
        "constraint) a candidate must have to be eligible for a preference pair.",
    )
    parser.add_argument(
        "--pairing-mode",
        choices=PAIRING_MODES,
        default="feasible_only",
        help="'feasible_only' (default): drop infeasible candidates, rank remaining pairs "
        "by objective score only (original behavior). 'lexicographic': keep infeasible "
        "candidates and rank feasibility ahead of objective score.",
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
        scored_sequences = load_scored_sequences(
            input_csv, reference_sequence, args.similarity_threshold, args.pairing_mode
        )
        all_pairs.extend(
            sample_pairs(
                scored_sequences=scored_sequences,
                reference_sequence=reference_sequence,
                num_pairs=args.pairs_per_input,
                rng=rng,
                pairing_mode=args.pairing_mode,
            )
        )

    write_csv(all_pairs, args.output_csv)
    write_jsonl(all_pairs, args.output_jsonl)
    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote JSONL: {args.output_jsonl}")
    print(f"Total DPO pairs: {len(all_pairs)}")


if __name__ == "__main__":
    main()
