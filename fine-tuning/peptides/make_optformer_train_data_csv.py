"""Builds OptFormer training data (paper's LLM-as-optimizer baseline):
history-conditioned windows sampled from real BO trajectories, target = the
sequence actually tried next in that trajectory. See
peptide_experiment/optformer.py::train_optformer and
peptide_experiment/optformer_optimization.py::run_optformer_bo for how this
feeds fine-tuning and held-out inference.

Score binning: OptFormer represents each trial's score as a discretized bin
index (paper: 1000 bins; this repo's much smaller per-milestone training
corpus uses fewer bins by default, see --num-bins) rather than a raw float,
so the model learns a coarse-grained, tokenizable notion of "how good" a
trial was. Bin edges are quantile cut points over the pooled feasible-score
distribution across every training task passed in -- computed once here,
frozen, and written to --bin-edges-output so held-out inference
(run_optformer_bo) reuses the exact same edges (recomputing them per-task
would make the model's learned bin semantics meaningless).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import importlib.util
import json
import random
from pathlib import Path

from make_train_data_csv import is_similar_enough

SCRIPT_DIR = Path(__file__).resolve().parent
REFSEQS_PATH = (
    SCRIPT_DIR / "../../optimization/peptides/apex_oracle/refseqs.py"
).resolve()

DEFAULT_NUM_BINS = 100
DEFAULT_CONTEXT_LENGTH = 100
DEFAULT_WINDOWS_PER_TASK = 50
DEFAULT_SIMILARITY_THRESHOLD = 0.75

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


def load_feasible_trajectory(
    input_csv: Path, reference_sequence: str, similarity_threshold: float
) -> list[tuple[str, float]]:
    """Feasible (sequence, score) rows from one task's trajectory CSV, in
    original chronological row order (duplicates kept -- a real trajectory
    can and does re-try the same sequence, and the training windows below
    are meant to reflect a real trial history, not a deduplicated candidate
    bank)."""
    rows: list[tuple[str, float]] = []
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
                continue
            try:
                score = float(row["train_y"])
            except ValueError:
                continue
            if not is_similar_enough(sequence, reference_sequence, similarity_threshold):
                continue
            rows.append((sequence, score))
    return rows


def compute_score_bin_edges(scores: list[float], num_bins: int) -> list[float]:
    """num_bins-1 interior quantile cut points splitting `scores` into
    num_bins equal-frequency bins (bin_for_score() bisects against these)."""
    if len(scores) < num_bins:
        raise ValueError(
            f"Need at least num_bins={num_bins} pooled feasible scores to compute bin edges, "
            f"got {len(scores)}"
        )
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    return [sorted_scores[min(n - 1, (i * n) // num_bins)] for i in range(1, num_bins)]


def bin_for_score(score: float, edges: list[float]) -> int:
    return bisect.bisect_right(edges, score)


def serialize_window(window: list[tuple[str, float]], edges: list[float]) -> str:
    return "\n".join(f"{seq} -> {bin_for_score(score, edges)}" for seq, score in window)


def build_windows(
    trajectories: dict[int, list[tuple[str, float]]],
    edges: list[float],
    context_length: int,
    windows_per_task: int,
    rng: random.Random,
) -> list[dict]:
    rows: list[dict] = []
    for task_label, traj in trajectories.items():
        if not traj:
            print(f"[make_optformer_train_data_csv] task {task_label}: 0 feasible trajectory rows, skipping")
            continue
        n = len(traj)
        offsets = rng.sample(range(n), min(windows_per_task, n))
        for s in offsets:
            context = traj[max(0, s - context_length):s]
            target_sequence, _target_score = traj[s]
            rows.append(
                {
                    "history_text": serialize_window(context, edges),
                    "target_sequence": target_sequence,
                }
            )
    return rows


def write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["history_text", "target_sequence"]
    with output_csv.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OptFormer training data (history-conditioned windows) from BOLT peptide trajectory CSVs."
    )
    parser.add_argument("--input-csv", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--reference-index",
        type=int,
        nargs="+",
        default=None,
        help="Index into apex_oracle/refseqs.py, one per --input-csv.",
    )
    parser.add_argument(
        "--reference-sequence",
        nargs="+",
        default=None,
        help="Overrides --reference-index; one per --input-csv.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bin-edges-output", type=Path, required=True)
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--windows-per-task", type=int, default=DEFAULT_WINDOWS_PER_TASK)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.reference_sequence is not None:
        reference_sequences = args.reference_sequence
    elif args.reference_index is not None:
        reference_sequences = [load_reference_sequence(i) for i in args.reference_index]
    else:
        raise ValueError("Must pass either --reference-index or --reference-sequence")

    if len(reference_sequences) != len(args.input_csv):
        raise ValueError(
            f"Expected one reference per input CSV, got {len(args.input_csv)} input CSVs "
            f"and {len(reference_sequences)} references."
        )
    task_labels = args.reference_index if args.reference_index is not None else list(range(len(args.input_csv)))

    trajectories: dict[int, list[tuple[str, float]]] = {}
    pooled_scores: list[float] = []
    for task_label, input_csv, reference_sequence in zip(task_labels, args.input_csv, reference_sequences):
        traj = load_feasible_trajectory(input_csv, reference_sequence, args.similarity_threshold)
        trajectories[task_label] = traj
        pooled_scores.extend(score for _seq, score in traj)
        print(f"[make_optformer_train_data_csv] task {task_label} ({input_csv}): {len(traj)} feasible trajectory rows")

    edges = compute_score_bin_edges(pooled_scores, args.num_bins)
    args.bin_edges_output.parent.mkdir(parents=True, exist_ok=True)
    args.bin_edges_output.write_text(json.dumps({"num_bins": args.num_bins, "edges": edges}, indent=2))
    print(f"Wrote {args.bin_edges_output} ({len(edges)} edges, {args.num_bins} bins)")

    rows = build_windows(trajectories, edges, args.context_length, args.windows_per_task, rng)
    write_csv(rows, args.output_csv)
    print(f"Wrote {args.output_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
