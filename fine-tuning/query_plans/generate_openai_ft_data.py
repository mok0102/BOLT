import argparse
import pandas as pd
import os
import json

import re


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a query-plan train CSV (task, train_x) into a chat JSONL."
    )
    parser.add_argument("--data-path", default="data/example_finetuning_data.csv")
    parser.add_argument("--save-path", default="data/example_finetuning_data.jsonl")
    return parser.parse_args()


# Additive: query_plan_experiment/trajectory_chain.py calls this with
# explicit --data-path/--save-path per milestone (the module-level hardcoded
# defaults below are otherwise unchanged, so any pre-existing direct-run
# usage of this script keeps working exactly as before).
_args = parse_args()
raw_data_path = _args.data_path
output_path = _args.save_path
excluded_tasks = ""
included_tasks = None


def extract_task_id(task):
    pattern = r"^.*[a-zA-Z]"
    match = re.search(pattern, task)
    if match:
        return match.group(0)
    else:
        return task


def get_task_description(task):
    # Note: This function assumes a 'workload' directory exists with SQL files.
    # Users should provide their own workload files or adjust the paths below.
    if not os.path.exists("./workload"):
        return "SELECT * FROM table"  # Placeholder return if workload dir is missing

    workload_group = task.split("_")[0].lower()
    task = task.split("_")[1].lower()

    if workload_group == "job":
        with open(f"./workload/job/{task}.sql") as f:
            return f.read()
    elif workload_group == "ceb":  # assume 3k for now, ignore 13k
        # get task id from task, everything before and including last alphabet character (a~z)
        task_id = extract_task_id(task)

        with open(f"./workload/ceb-3k/{task_id}/{task}.sql") as f:
            return f.read()


sample_prompt = {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides efficient orderings for given queries.",
        },
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
    ]
}


all_tasks = pd.read_csv(raw_data_path)
if excluded_tasks != "" and excluded_tasks[-3:] == "txt":
    excluded_task_list = pd.read_csv(excluded_tasks, header=None)[0].tolist()
    print(f"Excluding tasks from {excluded_tasks}")
elif excluded_tasks != "" and excluded_tasks[-3:] == "csv":
    excluded_task_list = pd.read_csv(excluded_tasks)["task"].tolist()
    excluded_task_list = list(set(excluded_task_list))
    print(f"excluding {len(excluded_task_list)} tasks")
    print(f"Excluding tasks from {excluded_tasks}")

# for each task, get the description, and fill description in the prompt as user content, and fill in train_x in the prompt as assistant content
train_jsonl = []
kept = 0
for i, row in all_tasks.iterrows():
    task = row["task"]
    if excluded_tasks != "" and task in excluded_task_list:
        print(f"Excluding task {task}")
        continue

    if included_tasks is not None:
        included_task_list = pd.read_csv(included_tasks, header=None)[0].tolist()
        if task not in included_task_list:
            print(f"Excluding task {task} which should not be included")
            continue
    kept += 1
    # if task[4:6] == '10' or task[4:6] == '11':
    #     continue
    # import pdb; pdb.set_trace()
    task_description = get_task_description(task)
    prompt = sample_prompt.copy()
    prompt["messages"][1]["content"] = task_description
    prompt["messages"][2]["content"] = row["train_x"]
    train_jsonl.append(json.dumps(prompt))


# save to jsonl
with open(output_path, "w") as f:
    # shuffle
    import random

    random.shuffle(train_jsonl)
    for line in train_jsonl:
        f.write(line + "\n")


from token_count import TokenCount

tc = TokenCount(model_name="gpt-4o-mini-2024-07-18")
# count tokens in the training data
tokens = tc.num_tokens_from_file(output_path)
print(f"Number of tokens in training data: {tokens}")
print(f"Kept {kept} tasks")
