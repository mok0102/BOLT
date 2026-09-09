import argparse
import ast
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "optimization/query_plans/query_plan_optimization"))
from query_plan_timing import span
from typing import Any, List, Optional

import pandas as pd
import torch
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a helpful assistant that provides efficient orderings for given queries."
)
BASE_DIR = Path(__file__).resolve().parent


def extract_task_id(task: str) -> str:
    pattern = r"^.*[a-zA-Z]"
    match = re.search(pattern, task)
    return match.group(0) if match else task


def get_task_description(task: str, workload_dir: Path) -> str:
    if not workload_dir.exists():
        return "SELECT * FROM table"

    workload_group = task.split("_")[0].lower()
    task_name = task.split("_")[1].lower()

    if workload_group == "job":
        with open(workload_dir / "job" / f"{task_name}.sql") as f:
            return f.read()

    if workload_group == "ceb":
        task_id = extract_task_id(task_name)
        with open(workload_dir / "ceb-3k" / task_id / f"{task_name}.sql") as f:
            return f.read()

    return "SELECT * FROM table"


def find_latest_hf_checkpoint(model_path: Path) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    index_names = {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
    if any((model_path / name).exists() for name in index_names):
        return model_path

    candidates = []
    for index_path in model_path.rglob("*"):
        if index_path.name not in index_names:
            continue
        parent = index_path.parent
        epoch_match = re.search(r"epoch_(\d+)$", parent.name)
        epoch = int(epoch_match.group(1)) if epoch_match else -1
        candidates.append((epoch, parent.stat().st_mtime, parent))

    if not candidates:
        raise FileNotFoundError(
            f"No HF checkpoint index found under {model_path}. "
            "Expected output/epoch_N/model.safetensors.index.json from torchtune."
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def parse_plan(text: str) -> Any:
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    # Keep only the first bracketed list if the model adds extra prose.
    match = re.search(r"\[[^\]]*\]", cleaned, flags=re.DOTALL)
    candidate = match.group(0) if match else cleaned

    try:
        parsed = ast.literal_eval(candidate)
    except (SyntaxError, ValueError):
        return cleaned

    if isinstance(parsed, list):
        return parsed
    return cleaned


def build_chat_prompt(tokenizer: Any, query: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{SYSTEM_PROMPT}\n\nUser query:\n{query}\n\nAnswer:"


def generate_for_query(
    model: Any,
    tokenizer: Any,
    query: str,
    num_samples: int,
    sample_batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> tuple[List[Any], List[str]]:
    prompt = build_chat_prompt(tokenizer, query)
    parsed_outputs: List[Any] = []
    raw_outputs: List[str] = []

    while len(raw_outputs) < num_samples:
        current_batch = min(sample_batch_size, num_samples - len(raw_outputs))
        prompts = [prompt] * current_batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_lengths = inputs.input_ids.shape[1]
        completions = generated_ids[:, prompt_lengths:]
        decoded = tokenizer.batch_decode(completions, skip_special_tokens=True)
        raw_outputs.extend(text.strip() for text in decoded)
        parsed_outputs.extend(parse_plan(text) for text in decoded)

    return parsed_outputs, raw_outputs


def load_tasks(tasks_file: Path, max_tasks: Optional[int]) -> List[str]:
    if tasks_file.exists():
        df = pd.read_csv(tasks_file)
        tasks = df["task"].dropna().unique().tolist()
    else:
        tasks = ["CEB_10A10"]

    if max_tasks is not None and max_tasks >= 0:
        tasks = tasks[:max_tasks]
    return tasks


def main(
    model_path: str = str(BASE_DIR / "output"),
    base_model_path: str = str(BASE_DIR / "ckpt" / "Qwen/Qwen2.5-3B-Instruct"),
    tasks_file: str = str(BASE_DIR / "data" / "example_finetuning_data.csv"),
    workload_dir: str = str(BASE_DIR / "workload"),
    output_file: str = str(BASE_DIR / "samples" / "sampled_plans_transformer.jsonl"),
    num_samples: int = 50,
    sample_batch_size: int = 4,
    max_tasks: Optional[int] = 5,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    dtype: str = "bf16",
) -> None:
    try:
        # This is a text-only Qwen sampler. In some environments transformers
        # tries to import torchvision while loading text models, and a broken
        # torch/torchvision pair can mask the real model import with
        # "Could not import module 'Qwen2ForCausalLM'".
        import transformers.utils.import_utils as import_utils

        import_utils._torchvision_available = False
        import_utils._torchvision_version = "N/A"
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "sampling_transformer.py requires transformers. Install it in this "
            "environment, e.g. `pip install transformers accelerate safetensors`."
        ) from exc

    model_dir = find_latest_hf_checkpoint(Path(model_path))
    base_dir = Path(base_model_path)
    print(f"Using tuned checkpoint: {model_dir}")
    print(f"Using tokenizer/config: {base_dir}")

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with span("llm_model_and_tokenizer_load"):
        tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(base_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()

    tasks = load_tasks(Path(tasks_file), max_tasks=max_tasks)
    task_descriptions = [
        get_task_description(task, Path(workload_dir)) for task in tasks
    ]

    results_df = pd.DataFrame(
        {
            "src_code": task_descriptions,
            "task": tasks,
            "generated_answers": [[] for _ in tasks],
            "generated_texts": [[] for _ in tasks],
        }
    )

    all_generated_answers = []
    all_generated_texts = []
    for task_description in tqdm(task_descriptions, total=len(task_descriptions)):
        with span("llm_generation", num_samples=num_samples):
            parsed_outputs, raw_outputs = generate_for_query(
                model=model,
                tokenizer=tokenizer,
                query=task_description,
                num_samples=num_samples,
                sample_batch_size=sample_batch_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
            )

        all_generated_answers.append(parsed_outputs)
        all_generated_texts.append(raw_outputs)

    results_df["generated_answers"] = all_generated_answers
    results_df["generated_texts"] = all_generated_texts

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_json(output_path, orient="records", lines=True)
    print(f"Saved samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(BASE_DIR / "output"))
    parser.add_argument(
        "--base-model-path",
        default=str(BASE_DIR / "ckpt" / "Qwen2.5-7B-Instruct"),
    )
    parser.add_argument(
        "--tasks-file",
        default=str(BASE_DIR / "data" / "example_finetuning_data.csv"),
    )
    parser.add_argument("--workload-dir", default=str(BASE_DIR / "workload"))
    parser.add_argument(
        "--output-file",
        default=str(BASE_DIR / "samples" / "sampled_plans_transformer.jsonl"),
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--max-tasks", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()
    main(**vars(args))
