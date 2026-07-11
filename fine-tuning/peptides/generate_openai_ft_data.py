import argparse
import pandas as pd
import ast
import json
import os

sample_prompt = {
    "messages": [
        {
            "role": "system",
            "content": f"You are a specialized assistant that modifies peptide sequences to enhance antimicrobial activity. Make up to 25% sequence modifications based on known antimicrobial peptide properties such as: positive charge, hydrophobicity, and amphipathicity.",
        },
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
    ]
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert peptide train CSV into OpenAI/torchtune chat JSONL."
    )
    parser.add_argument("--data-path", default="./train_data/train_data.csv")
    parser.add_argument("--save-path", default="./train_data/train_data.jsonl")
    return parser.parse_args()


args = parse_args()

# Placeholders
save_path = args.save_path
data_path = args.data_path

# Ensure directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

if not os.path.exists(data_path):
    print(
        f"Please provide a csv file at {data_path} with columns 'sequence' and 'reference_sequence'"
    )
    # Create dummy data for demonstration if file doesn't exist
    data = pd.DataFrame(
        {
            "sequence": ["RRTYFQLEQASRKGNRGFRR", "RRYYEQLEFASRKVNRGFRA"],
            "reference_sequence": ["RRYYEQLEQASRKGNRGFRR", "RRYYEQLEQASRKGNRGFRR"],
        }
    )
else:
    data = pd.read_csv(data_path)

train_jsonl = []
for i, row in data.iterrows():
    target_sequence = row["sequence"]
    source_sequence = row["reference_sequence"]

    prompt = sample_prompt.copy()
    prompt["messages"][1]["content"] = source_sequence
    prompt["messages"][2]["content"] = target_sequence

    train_jsonl.append(json.dumps(prompt))

with open(save_path, "w") as f:
    f.write("\n".join(train_jsonl))

try:
    from token_count import TokenCount

    tc = TokenCount(model_name="gpt-4o-mini-2024-07-18")
    tokens = tc.num_tokens_from_file(save_path)
    print(f"Number of tokens in training data: {tokens}")
except ImportError:
    print("token_count module not found. Skipping token count.")
