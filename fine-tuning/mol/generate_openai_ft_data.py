"""Mol analogue of fine-tuning/peptides/generate_openai_ft_data.py: converts
make_train_data_csv.py's output CSV into the chat JSONL torchtune's
chat_dataset consumes (read-only reference to that script, never imported
-- isolation contract). Builds messages via optimization/mol/mol_prompt.py's
own SYSTEM_PROMPT/make_messages -- the exact wording used everywhere else a
mol prompt is built (sampling, reference-model scoring) -- instead of
duplicating prompt text here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "optimization" / "mol"))
from mol_prompt import make_messages  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert mol train CSV into torchtune chat JSONL.")
    parser.add_argument("--data-path", type=Path, default=Path("./train_data/train_data.csv"))
    parser.add_argument("--save-path", type=Path, default=Path("./train_data/train_data.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Expected a csv file at {args.data_path} with columns 'candidate_selfies', "
            "'target_sequence', 'seed_selfies' (built by make_train_data_csv.py)"
        )

    lines = []
    with args.data_path.open(newline="") as f_in:
        reader = csv.DictReader(f_in)
        required_columns = {"candidate_selfies", "target_sequence", "seed_selfies"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Expected columns {sorted(required_columns)} in {args.data_path}, missing {sorted(missing_columns)}")
        for row in reader:
            messages = make_messages(row["target_sequence"], row["seed_selfies"], row["candidate_selfies"])
            lines.append(json.dumps({"messages": messages}))

    args.save_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"Wrote JSONL: {args.save_path} ({len(lines)} rows)")


if __name__ == "__main__":
    main()
