"""Create a JSONL of individually-sampled, similarity-infeasible peptide
completions for the "dpo_ii" ablation's auxiliary infeasibility-suppression
term (see fine-tuning/peptides/dpo_ii/loss.py).

Unlike make_dpo_train_data_csv.py's --pairing-mode feasibility_aware (which
pairs two infeasible candidates together for FeasibilityAwareORPTLoss's l_ii
term), this script does not pair anything: l_ii suppresses each side's
u = log pi_theta - log pi_ref independently (no cross term between the two
sides of a pair), so pairing two infeasible sequences together was only ever
a data-plumbing convenience for that loss, not a mathematical requirement.
Here we just sample individual infeasible completions (with replacement --
no need for >=2 *distinct* infeasible candidates in the pool) and write them
out as single-sequence chat examples, in the same {"messages": [...]}
"openai"-style format torchtune.datasets.chat_dataset already reads for the
BOLT SFT stage.
"""

import argparse
import json
import random
from pathlib import Path

from make_dpo_train_data_csv import (
    REFERENCE_SEQUENCE,
    load_reference_sequence,
    load_scored_sequences,
    make_messages,
)

DEFAULT_NUM_SINGLES = 1000
SIMILARITY_THRESHOLD = 0.75

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = (
    SCRIPT_DIR
    / "../../optimization/peptides/lolbo_scripts/optimization_all_collected_data/"
    / "BOLT-apex_no-wandb-tracking_all-data-collected.csv"
).resolve()
DEFAULT_OUTPUT_JSONL = SCRIPT_DIR / "train_data" / "infeasible_singles.jsonl"


def sample_infeasible_singles(
    scored_sequences: list[tuple[str, float, bool]],
    reference_sequence: str,
    num_samples: int,
    rng: random.Random,
) -> list[dict]:
    """Returns num_samples {"messages": [...]} rows drawn from the infeasible
    subset of scored_sequences, sampled independently with replacement."""
    infeasible = [s for s in scored_sequences if not s[2]]
    if not infeasible:
        raise ValueError(
            "No infeasible sequences available to build infeasible-singles "
            f"data (0 of {len(scored_sequences)} candidates fail the similarity constraint)."
        )
    picks = [rng.choice(infeasible) for _ in range(num_samples)]
    return [{"messages": make_messages(reference_sequence, sequence)} for sequence, _score, _feasible in picks]


def write_jsonl(rows: list[dict], output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as f_out:
        for row in rows:
            f_out.write(json.dumps(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create individually-sampled infeasible-peptide chat examples for the dpo_ii ablation."
    )
    parser.add_argument("--input-csv", type=Path, nargs="+", default=[DEFAULT_INPUT_CSV])
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
        "--singles-per-input",
        type=int,
        default=DEFAULT_NUM_SINGLES,
        help="Infeasible completions to sample (with replacement) from each input CSV.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help="Minimum similarity to the reference sequence a candidate must have to count as feasible "
        "(matches the BO similarity constraint) -- candidates below this are the infeasible pool sampled here.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.reference_index is not None:
        reference_sequences = [load_reference_sequence(reference_index) for reference_index in args.reference_index]
    else:
        reference_sequences = [args.reference_sequence or REFERENCE_SEQUENCE] * len(args.input_csv)

    if len(args.input_csv) != len(reference_sequences):
        raise ValueError(
            f"Expected one reference per input CSV, got {len(args.input_csv)} "
            f"input CSVs and {len(reference_sequences)} references."
        )

    all_rows = []
    for input_csv, reference_sequence in zip(args.input_csv, reference_sequences):
        # pairing_mode="lexicographic" is used purely for its
        # load_scored_sequences() side effect of keeping infeasible rows
        # instead of dropping them (see that function's docstring) --
        # no pairing actually happens here.
        scored_sequences = load_scored_sequences(
            input_csv, reference_sequence, args.similarity_threshold, pairing_mode="lexicographic"
        )
        all_rows.extend(
            sample_infeasible_singles(
                scored_sequences=scored_sequences,
                reference_sequence=reference_sequence,
                num_samples=args.singles_per_input,
                rng=rng,
            )
        )

    write_jsonl(all_rows, args.output_jsonl)
    print(f"Wrote JSONL: {args.output_jsonl}")
    print(f"Total infeasible singles: {len(all_rows)}")


if __name__ == "__main__":
    main()
