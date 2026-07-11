import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_JSONL = [SCRIPT_DIR / "sample_output_transformers.jsonl"]
DEFAULT_OUTPUT_INIT = SCRIPT_DIR / "seed_0_init.txt"
DEFAULT_OUTPUT_SCORES = SCRIPT_DIR / "seed_0_scores.csv"

OPTIMIZATION_PEPTIDES_DIR = (
    SCRIPT_DIR / "../../../optimization/peptides"
).resolve()
AMINO_ACID_PATTERN = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert sampled transformer output JSONL into BOLT initialization "
            "files matching seed_0_init.txt and seed_0_scores.csv."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        nargs="+",
        default=DEFAULT_INPUT_JSONL,
        help="One or more sampled transformer JSONL files to merge in order.",
    )
    parser.add_argument("--output-init", type=Path, default=DEFAULT_OUTPUT_INIT)
    parser.add_argument("--output-scores", type=Path, default=DEFAULT_OUTPUT_SCORES)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Keep only the first occurrence of each generated sequence.",
    )
    parser.add_argument(
        "--skip-scores",
        action="store_true",
        help="Only write the init sequence file; do not run the APEX oracle.",
    )
    return parser.parse_args()


def extract_sequence(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None

    if AMINO_ACID_PATTERN.fullmatch(text):
        return text

    matches = AMINO_ACID_PATTERN.findall(text)
    if not matches:
        return None
    return max(matches, key=len)


def iter_generated_answers(input_jsonl: Path):
    with input_jsonl.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            if "generated_answers" in record:
                answers = record["generated_answers"]
            elif "messages" in record:
                answers = [
                    message["content"]
                    for message in record["messages"]
                    if message.get("role") == "assistant"
                ]
            else:
                raise ValueError(
                    f"Unsupported JSONL record at line {line_number}: {record.keys()}"
                )

            for answer in answers:
                if answer is None:
                    continue
                sequence = extract_sequence(str(answer))
                if sequence:
                    yield sequence


def load_sequences(input_jsonls: list[Path], deduplicate: bool) -> list[str]:
    sequences = []
    seen = set()

    for input_jsonl in input_jsonls:
        for sequence in iter_generated_answers(input_jsonl):
            if deduplicate:
                if sequence in seen:
                    continue
                seen.add(sequence)
            sequences.append(sequence)

    if not sequences:
        input_paths = ", ".join(str(path) for path in input_jsonls)
        raise ValueError(f"No generated peptide sequences found in {input_paths}")
    return sequences


def write_sequences(sequences: list[str], output_init: Path) -> None:
    output_init.parent.mkdir(parents=True, exist_ok=True)
    with output_init.open("w") as f:
        for sequence in sequences:
            f.write(sequence + "\n")


def score_sequences(sequences: list[str], chunk_size: int) -> np.ndarray:
    sys.path.append(str(OPTIMIZATION_PEPTIDES_DIR))
    from apex_oracle import apex_wrapper

    scores = []
    for start in tqdm(range(0, len(sequences), chunk_size), desc="Scoring sequences"):
        chunk = sequences[start : start + chunk_size]
        apex_scores = apex_wrapper(chunk)
        scores.append(-apex_scores[:, 0])
    return np.hstack(scores)


def write_scores(scores: np.ndarray, output_scores: Path) -> None:
    output_scores.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_scores, scores, delimiter=",", fmt="%.8f")


def main() -> None:
    args = parse_args()

    sequences = load_sequences(args.input_jsonl, deduplicate=args.deduplicate)
    write_sequences(sequences, args.output_init)
    print(f"Wrote init sequences: {args.output_init}")
    print(f"Number of sequences: {len(sequences)}")

    if args.skip_scores:
        print("Skipping score generation.")
        return

    scores = score_sequences(sequences, chunk_size=args.chunk_size)
    write_scores(scores, args.output_scores)
    print(f"Wrote scores: {args.output_scores}")


if __name__ == "__main__":
    main()
