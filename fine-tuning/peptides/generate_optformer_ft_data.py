"""Converts make_optformer_train_data_csv.py's history_text/target_sequence
CSV into a torchtune chat JSONL (torchtune.datasets.chat_dataset,
conversation_column=messages, conversation_style=openai -- same schema
generate_openai_ft_data.py already produces for BOLT-SFT, just with
OptFormer's history-conditioned prompt instead of BOLT-SFT's plain
reference-sequence prompt).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

NO_HISTORY_PLACEHOLDER = "(no trials yet)"


def system_prompt(num_bins: int) -> str:
    return (
        "You are a Bayesian optimization assistant proposing antimicrobial peptide "
        "sequences. You will be shown a history of previously tried sequences, each "
        f"labeled with its score bin (0 = worst, {num_bins - 1} = best, out of {num_bins} bins). "
        "Propose one new peptide sequence expected to score in a higher bin than any "
        "shown. Respond with only the sequence."
    )


def make_messages(history_text: str, target_sequence: str, num_bins: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt(num_bins)},
        {"role": "user", "content": history_text.strip() or NO_HISTORY_PLACEHOLDER},
        {"role": "assistant", "content": target_sequence},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OptFormer history/target CSV into torchtune chat JSONL."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument(
        "--num-bins",
        type=int,
        required=True,
        help="Must match make_optformer_train_data_csv.py's --num-bins for this data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with args.data_path.open(newline="") as f_in, args.save_path.open("w") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            messages = make_messages(row["history_text"], row["target_sequence"], args.num_bins)
            f_out.write(json.dumps({"messages": messages}) + "\n")
            n_rows += 1
    print(f"Wrote {args.save_path} ({n_rows} rows)")


if __name__ == "__main__":
    main()
