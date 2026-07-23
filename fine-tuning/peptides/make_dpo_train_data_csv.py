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

# (mixed, both_feasible, both_infeasible) draw-probability weights for
# --pairing-mode feasibility_aware's sample_pairs(), renormalized over
# whichever of the three types the pool can actually supply. Equal by
# default -- see sample_pairs()'s feasibility_aware branch for why this
# should NOT default to the raw pool's feasible/infeasible combinatorics.
DEFAULT_FEASIBILITY_AWARE_PAIR_TYPE_MIX = (1.0, 1.0, 1.0)

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


PAIRING_MODES = ("feasible_only", "lexicographic", "feasibility_aware")


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

    pairing_mode="feasibility_aware": same as "lexicographic" -- infeasible
    rows are kept, not dropped. Unlike "lexicographic", sample_pairs() for
    this mode does not skip both-infeasible draws (see its docstring); the
    feasibility_aware ORPT loss needs both-infeasible pairs to downweight
    both candidates, whereas lexicographic pairing treats them as carrying
    no signal.
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

    kept_or_dropped = "dropped" if pairing_mode == "feasible_only" else "kept for ranking"
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
) -> tuple[str, float, bool, str, float, bool] | None:
    """Returns (chosen_seq, chosen_score, chosen_feasible, rejected_seq,
    rejected_score, rejected_feasible) for this sampled candidate pair, or
    None if it carries no training signal.

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

    pairing_mode="feasibility_aware": same mixed-pair rule as
    "lexicographic" (feasible side always wins). Unlike "lexicographic",
    both-infeasible pairs are NOT skipped -- they're returned with an
    arbitrary (first-as-chosen) assignment, since the feasibility_aware
    ORPT loss's infeasible-vs-infeasible term is symmetric in the two sides
    and doesn't use the objective score at all for that branch.
    """
    first_sequence, first_score, first_feasible = first
    second_sequence, second_score, second_feasible = second

    feasibility_ranked = pairing_mode in ("lexicographic", "feasibility_aware")

    if feasibility_ranked and first_feasible != second_feasible:
        if first_feasible:
            return first_sequence, first_score, True, second_sequence, second_score, False
        return second_sequence, second_score, True, first_sequence, first_score, False

    if pairing_mode == "lexicographic" and not first_feasible and not second_feasible:
        return None

    if pairing_mode == "feasibility_aware" and not first_feasible and not second_feasible:
        return first_sequence, first_score, False, second_sequence, second_score, False

    if first_score == second_score:
        return None
    if first_score > second_score:
        return first_sequence, first_score, first_feasible, second_sequence, second_score, second_feasible
    return second_sequence, second_score, second_feasible, first_sequence, first_score, first_feasible


def _build_pair(reference_sequence: str, result: tuple[str, float, bool, str, float, bool]) -> dict:
    (
        chosen_sequence,
        chosen_score,
        chosen_feasible,
        rejected_sequence,
        rejected_score,
        rejected_feasible,
    ) = result
    return {
        "reference_sequence": reference_sequence,
        "chosen_sequence": chosen_sequence,
        "rejected_sequence": rejected_sequence,
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "chosen_feasible": chosen_feasible,
        "rejected_feasible": rejected_feasible,
        "chosen": make_messages(reference_sequence, chosen_sequence),
        "rejected": make_messages(reference_sequence, rejected_sequence),
    }


def sample_pairs(
    scored_sequences: list[tuple[str, float, bool]],
    reference_sequence: str,
    num_pairs: int,
    rng: random.Random,
    pairing_mode: str = "feasible_only",
    pair_type_weights: tuple[float, float, float] | None = None,
) -> list[dict]:
    """pair_type_weights (mixed, both_feasible, both_infeasible): only used
    by pairing_mode="feasibility_aware" -- see its branch below for why this
    doesn't default to the raw pool's combinatorics."""
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

    if pairing_mode == "feasibility_aware":
        feasible = [s for s in scored_sequences if s[2]]
        infeasible = [s for s in scored_sequences if not s[2]]
        if not feasible:
            raise ValueError(
                "No feasible sequences available for feasibility_aware pairing "
                f"(0 of {len(scored_sequences)} candidates satisfy the similarity "
                "constraint) -- every pair needs a feasible side to supply either "
                "the feasible-vs-feasible ranking signal or the 'infeasible loses' signal."
            )

        # Unlike the "lexicographic" branch above (which weights mixed vs
        # both-feasible draws by how many of each actually exist, matching
        # what a uniform draw over the whole pool would produce), this does
        # NOT sample proportional to the raw pool's feasible/infeasible
        # combinatorics: on real BO trajectory data only ~5-10% of candidates
        # are feasible, so a combinatorics-proportional draw is >90%
        # both-infeasible and <1% both-feasible (confirmed on real data:
        # 0.4% ff / 6.5% fi / 93.1% ii out of 14000 pairs at milestone 14 of
        # peptide_100task_orpt_fa) -- starving the feasible-vs-feasible
        # ranking term of almost all its training signal, which directly
        # undermines this loss's whole point (see fa_orpt/loss.py's
        # docstring: the DPO-style loss's failure mode is specifically about
        # feasible-vs-feasible ranking not getting a meaningful margin).
        # Instead, target an even three-way mix by default (pair_type_weights),
        # falling back to whichever of the three types the pool can actually
        # supply (both-feasible needs >=2 feasible candidates, both-infeasible
        # needs >=2 infeasible candidates; a pair type with zero available
        # weight has its share redistributed across the remaining types).
        if pair_type_weights is None:
            pair_type_weights = DEFAULT_FEASIBILITY_AWARE_PAIR_TYPE_MIX
        mixed_weight, both_feasible_weight, both_infeasible_weight = pair_type_weights
        can_mixed = bool(infeasible)
        can_both_feasible = len(feasible) >= 2
        can_both_infeasible = len(infeasible) >= 2

        total_weight = (
            (mixed_weight if can_mixed else 0.0)
            + (both_feasible_weight if can_both_feasible else 0.0)
            + (both_infeasible_weight if can_both_infeasible else 0.0)
        )
        if total_weight <= 0:
            raise ValueError(
                f"Need at least 2 usable scored sequences to pair, got {len(scored_sequences)}."
            )
        mixed_probability = (mixed_weight / total_weight) if can_mixed else 0.0
        both_feasible_probability = (both_feasible_weight / total_weight) if can_both_feasible else 0.0

        pairs = []
        attempts = 0
        while len(pairs) < num_pairs and attempts < max_attempts:
            attempts += 1
            draw = rng.random()
            if can_mixed and draw < mixed_probability:
                first, second = rng.choice(feasible), rng.choice(infeasible)
            elif can_both_feasible and draw < mixed_probability + both_feasible_probability:
                first, second = rng.sample(feasible, 2)
            elif can_both_infeasible:
                first, second = rng.sample(infeasible, 2)
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
        "chosen_feasible",
        "rejected_feasible",
    ]
    with output_csv.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    fieldname: (
                        int(pair[fieldname])
                        if fieldname in ("chosen_feasible", "rejected_feasible")
                        else pair[fieldname]
                    )
                    for fieldname in fieldnames
                }
            )


def write_jsonl(pairs: list[dict], output_jsonl: Path) -> None:
    """Extra keys beyond "chosen"/"rejected" (chosen_feasible,
    rejected_feasible) are ignored by torchtune's stock
    preference_dataset/PreferenceDataset, which only reads "chosen" and
    "rejected" -- so this is safe for the existing DPO ("dpo") loss path.
    fa_orpt.dataset.FeasibilityAwarePreferenceDataset is what actually reads
    the feasibility fields back out.
    """
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as f_out:
        for pair in pairs:
            f_out.write(
                json.dumps(
                    {
                        "chosen": pair["chosen"],
                        "rejected": pair["rejected"],
                        "chosen_feasible": int(pair["chosen_feasible"]),
                        "rejected_feasible": int(pair["rejected_feasible"]),
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
        "candidates and rank feasibility ahead of objective score, skipping both-infeasible "
        "draws (no signal). 'feasibility_aware': like 'lexicographic', but also keeps "
        "both-infeasible pairs (for the feasibility_aware ORPT loss's infeasible-vs-infeasible "
        "term) instead of skipping them.",
    )
    parser.add_argument(
        "--feasibility-aware-mix",
        type=float,
        nargs=3,
        metavar=("MIXED_WEIGHT", "BOTH_FEASIBLE_WEIGHT", "BOTH_INFEASIBLE_WEIGHT"),
        default=list(DEFAULT_FEASIBILITY_AWARE_PAIR_TYPE_MIX),
        help="Only used by --pairing-mode feasibility_aware: relative draw weights for "
        "(mixed, both-feasible, both-infeasible) pair types, renormalized over whichever "
        "types the pool can actually supply. Default is an even 1:1:1 mix -- NOT "
        "proportional to the raw pool's feasible/infeasible combinatorics, since real BO "
        "trajectory data is only ~5-10%% feasible and a combinatorics-proportional draw "
        "would starve the feasible-vs-feasible ranking term of training signal.",
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
                pair_type_weights=tuple(args.feasibility_aware_mix),
            )
        )

    write_csv(all_pairs, args.output_csv)
    write_jsonl(all_pairs, args.output_jsonl)
    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote JSONL: {args.output_jsonl}")
    print(f"Total DPO pairs: {len(all_pairs)}")


if __name__ == "__main__":
    main()
