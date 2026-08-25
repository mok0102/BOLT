"""Held-out inference for the OptFormer baseline (paper's LLM-as-optimizer
comparison): a from-scratch, history-conditioned propose/score/append loop
using a fine-tuned OptFormer-<milestone> checkpoint (peptide_experiment/
optformer.py::train_optformer) -- no GP/BoTorch surrogate or acquisition
function at all, unlike every other arm's steps.py::run_bo(). Produces the
same train_x,train_y-column CSV contract run_bo() does, so it slots into the
same aggregate.py/heldout_eval.py pipeline unchanged. Standalone (not a
steps.py flag): shares no code with the GP/BoTorch inner loop.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from .config import ExperimentConfig
from .steps import build_mutation_init

BOLT_ROOT = Path(__file__).resolve().parents[1]
FINE_TUNING_DIR = BOLT_ROOT / "fine-tuning" / "peptides"
if str(FINE_TUNING_DIR) not in sys.path:
    sys.path.insert(0, str(FINE_TUNING_DIR))

from generate_optformer_ft_data import system_prompt  # noqa: E402
from make_optformer_train_data_csv import serialize_window  # noqa: E402
from sampling_transformers import clean_generation, resolve_device, resolve_dtype  # noqa: E402

MAX_NEW_TOKENS = 64
TOP_P = 0.95


def _load_model_and_tokenizer(checkpoint_path: Path, device: str, dtype):
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    except ValueError:
        tokenizer_json = Path(checkpoint_path) / "tokenizer.json"
        if not tokenizer_json.exists():
            raise
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_json),
            eos_token="<|im_end|>",
            pad_token="<|endoftext|>",
            additional_special_tokens=["<|im_start|>", "<|im_end|>"],
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model, tokenizer


def _build_prompt(tokenizer, history_text: str, num_bins: int) -> str:
    messages = [
        {"role": "system", "content": system_prompt(num_bins)},
        {"role": "user", "content": history_text.strip() or "(no trials yet)"},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{messages[0]['content']}\n\nUser: {messages[1]['content']}\nAssistant:"


def _generate_candidates(model, tokenizer, prompt: str, bsz: int, temperature: float, device: str) -> list[str]:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = encoded["input_ids"].shape[-1]
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=bsz,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    answers = []
    for sequence in generated:
        new_tokens = sequence[input_length:]
        answers.append(clean_generation(tokenizer.decode(new_tokens, skip_special_tokens=True)))
    return answers


def run_optformer_bo(
    cfg: ExperimentConfig,
    task_idx: int,
    work_dir: Path,
    run_id: str,
    milestone: int,
    checkpoint_path: Path,
) -> Path:
    """Self-seeds via steps.build_mutation_init() (OptFormer has no
    task-specific proposer any more than STBO does -- its checkpoint only
    drives subsequent proposals), then repeatedly prompts the fine-tuned
    checkpoint with the running trial history (serialized via the SAME
    frozen score-bin edges written at training time -- never recomputed
    per-task, or the model's learned bin semantics become meaningless) to
    propose cfg.bsz new candidates per round, scores them with the real APEX
    oracle, and appends unconditionally (no feasibility post-filtering,
    matching how LOLBO's own collected-data CSV already includes infeasible
    rows) until cfg.oracle_budget calls have been made (init pool counts
    toward the budget, same convention run_bo()'s init_size baseline uses).
    """
    from apex_oracle import apex_wrapper

    work_dir.mkdir(parents=True, exist_ok=True)
    dest_csv = work_dir / f"task_{task_idx:04d}.csv"
    if dest_csv.exists():
        print(f"[{run_id} task {task_idx}] trajectory already exists at {dest_csv}, skipping")
        return dest_csv

    init_path, scores_path = build_mutation_init(cfg, task_idx, work_dir)
    seqs = [line for line in init_path.read_text().splitlines() if line.strip()]
    scores = [float(line) for line in scores_path.read_text().splitlines() if line.strip()]
    history: list[tuple[str, float]] = list(zip(seqs, scores))

    bin_edges_path = cfg.optformer_dir / f"score_bin_edges_{milestone}.json"
    bin_meta = json.loads(bin_edges_path.read_text())
    edges, num_bins = bin_meta["edges"], bin_meta["num_bins"]

    device = resolve_device("auto")
    dtype = resolve_dtype("auto", device)
    model, tokenizer = _load_model_and_tokenizer(checkpoint_path, device, dtype)

    # cfg.oracle_budget counts calls made AFTER the init pool, matching
    # steps.run_bo()'s --max_n_oracle_calls convention (init_size + oracle_budget
    # rows total) -- confirmed against a real MTBO smoke run: init_size=10/
    # oracle_budget=20 produced 30 rows. n_extra_calls tracks only the
    # post-init calls, so every arm consumes the same real oracle budget for
    # a fair comparison.
    n_extra_calls = 0
    max_rounds = 2 * -(-cfg.oracle_budget // max(cfg.bsz, 1))  # ceil-div safety cap against an all-empty-generation loop
    rounds = 0
    while n_extra_calls < cfg.oracle_budget and rounds < max_rounds:
        rounds += 1
        recent = history[-cfg.optformer_context_length:]
        prompt = _build_prompt(tokenizer, serialize_window(recent, edges), num_bins)
        batch_size = min(cfg.bsz, cfg.oracle_budget - n_extra_calls)
        candidates = [c for c in _generate_candidates(model, tokenizer, prompt, batch_size, cfg.optformer_temperature, device) if c]
        if not candidates:
            print(f"[{run_id} task {task_idx}] round {rounds}: all-empty generation, retrying")
            continue
        candidate_scores = list(-apex_wrapper(candidates)[:, 0])
        history.extend(zip(candidates, candidate_scores))
        n_extra_calls += len(candidates)

    if n_extra_calls < cfg.oracle_budget:
        print(
            f"[{run_id} task {task_idx}] giving up after {rounds} rounds with only "
            f"{n_extra_calls}/{cfg.oracle_budget} post-init oracle calls made (persistent all-empty generations)"
        )

    with dest_csv.open("w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["train_x", "train_y"])
        writer.writerows(history)
    return dest_csv
