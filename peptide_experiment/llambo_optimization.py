"""Held-out inference for the LLAMBO baseline (Liu et al. 2024):
zero-shot, in-context Bayesian optimization using three prompted roles
against the same unmodified base_checkpoint_dir every other arm fine-tunes
from -- no fine-tuning stage at all (paper's own LLAMBO uses GPT-4o-mini
"out-of-the-box"). Mirrors optformer_optimization.py's shape: standalone,
history-conditioned propose/score/append loop producing the same
train_x,train_y CSV contract steps.run_bo() does.

Three roles, each a system+user chat prompt (mirroring
optformer_optimization.py::_build_prompt): (1) unconditioned candidate
sampler -- propose a sequence expected to beat everything shown; (2)
target-conditioned sampler -- propose a sequence expected to hit a specific
target score; (3) in-context surrogate -- predict a candidate's score given
history, sampled llambo_k_mc_samples times for an empirical mean/std.

Score representation: fresh min-max normalization to an integer 0-100 over
the currently-visible window, recomputed every round -- unlike OptFormer's
frozen training-time bin edges, LLAMBO has no training phase to freeze
anything from, and integer normalization is far more parseable by a
non-fine-tuned 3B model than raw floats. Expected improvement is computed
in this same normalized [0, 1] fraction space (mu/sigma/best all consistent
fractions of the visible window's score range) for scale-invariant ranking;
this is a reasonable, non-verbatim reimplementation of the paper's EI
formula, not a verbatim reproduction.

Token-budget accounting: input tokens are counted once per unique prompt
submitted (not multiplied by num_return_sequences), matching real API
billing semantics. The loop terminates once oracle_budget real calls are
made OR the token budget is exhausted -- early termination from token
exhaustion is expected and correct (the original paper's own LLAMBO run
"completed fewer than 100 optimization steps" for the same reason), not a
bug to avoid.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExperimentConfig
from .steps import build_mutation_init

BOLT_ROOT = Path(__file__).resolve().parents[1]
FINE_TUNING_DIR = BOLT_ROOT / "fine-tuning" / "peptides"
if str(FINE_TUNING_DIR) not in sys.path:
    sys.path.insert(0, str(FINE_TUNING_DIR))

from sampling_transformers import clean_generation, resolve_device, resolve_dtype  # noqa: E402

MAX_NEW_TOKENS = 64
SURROGATE_MAX_NEW_TOKENS = 8
INT_RE = re.compile(r"-?\d+")


def _load_model_and_tokenizer(checkpoint_path: Path, device: str, dtype):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model, tokenizer


def _normalize_scores(scores: list[float]) -> list[int]:
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [50 for _ in scores]
    return [round((s - lo) / (hi - lo) * 100) for s in scores]


def _serialize_window(window: list[tuple[str, float]]) -> tuple[str, list[int]]:
    normalized = _normalize_scores([score for _seq, score in window])
    text = "\n".join(f"{seq} -> {n}" for (seq, _score), n in zip(window, normalized))
    return text, normalized


def _candidate_system_prompt(target: int | None) -> str:
    base = (
        "You are a Bayesian optimization assistant proposing antimicrobial peptide "
        "sequences. You will be shown a history of previously tried sequences, each "
        "labeled with its normalized score (0 = worst seen, 100 = best seen). "
    )
    if target is None:
        return base + "Propose one new peptide sequence expected to score higher than any shown. Respond with only the sequence, nothing else."
    return base + (
        f"Propose one new peptide sequence expected to achieve a normalized score of "
        f"exactly {target}. Respond with only the sequence, nothing else."
    )


def _surrogate_system_prompt() -> str:
    return (
        "You are a Bayesian optimization surrogate model predicting antimicrobial "
        "peptide scores. You will be shown a history of previously tried sequences, "
        "each labeled with its normalized score (0 = worst seen, 100 = best seen), "
        "followed by one candidate sequence. Respond with only your predicted "
        "normalized score for the candidate, as a single integer from 0 to 100, "
        "nothing else."
    )


def _build_prompt(tokenizer, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user.strip() or "(no trials yet)"}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\nUser: {user}\nAssistant:"


def _generate(
    model, tokenizer, prompt: str, num_return_sequences: int, temperature: float, top_p: float,
    max_new_tokens: int, device: str,
) -> tuple[list[str], int]:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = encoded["input_ids"].shape[-1]
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    outputs = [clean_generation(tokenizer.decode(seq[input_length:], skip_special_tokens=True)) for seq in generated]
    return outputs, input_length


def _parse_int_0_100(text: str) -> int | None:
    match = INT_RE.search(text)
    if match is None:
        return None
    return max(0, min(100, int(match.group())))


def _expected_improvement(mu: float, sigma: float, best: float, xi: float) -> float:
    margin = mu - best - xi
    if sigma <= 1e-9:
        return max(margin, 0.0)
    z = margin / sigma
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return margin * cdf + sigma * pdf


def run_llambo_bo(
    cfg: ExperimentConfig,
    task_idx: int,
    work_dir: Path,
    run_id: str,
    checkpoint_path: Path | None = None,
) -> Path:
    """Self-seeds via steps.build_mutation_init() (same as OptFormer -- LLAMBO
    has no task-specific proposer of its own either), then runs LLAMBO's
    sequential propose/rank/score loop against the unmodified base checkpoint
    (checkpoint_path=None -> cfg.base_checkpoint_dir; LLAMBO is never
    fine-tuned, so there's no per-milestone checkpoint to pass here) until
    cfg.oracle_budget real oracle calls are made or the token budget is
    exhausted. Writes a task_XXXX_llambo_meta.json sidecar recording how the
    run ended, so a token-exhausted short CSV is distinguishable from a real
    failure -- aggregate.py::best_mic_at_k()'s existing
    row_idx = min(init_size+k, len(df)) clamp already tolerates a
    shorter-than-expected CSV gracefully.
    """
    from apex_oracle import apex_wrapper

    work_dir.mkdir(parents=True, exist_ok=True)
    dest_csv = work_dir / f"task_{task_idx:04d}.csv"
    meta_path = work_dir / f"task_{task_idx:04d}_llambo_meta.json"
    if dest_csv.exists():
        print(f"[{run_id} task {task_idx}] trajectory already exists at {dest_csv}, skipping")
        return dest_csv

    init_path, scores_path = build_mutation_init(cfg, task_idx, work_dir)
    seqs = [line for line in init_path.read_text().splitlines() if line.strip()]
    scores = [float(line) for line in scores_path.read_text().splitlines() if line.strip()]
    history: list[tuple[str, float]] = list(zip(seqs, scores))

    checkpoint_path = checkpoint_path or cfg.base_checkpoint_dir
    device = resolve_device("auto")
    dtype = resolve_dtype("auto", device)
    model, tokenizer = _load_model_and_tokenizer(checkpoint_path, device, dtype)

    n_extra_calls = 0
    total_input_tokens = 0
    rounds = 0
    max_rounds = 2 * -(-cfg.oracle_budget // max(cfg.llambo_batch_size, 1))  # ceil-div safety cap, mirrors run_optformer_bo

    while True:
        if n_extra_calls >= cfg.oracle_budget:
            terminated_reason = "oracle_budget_reached"
            break
        if total_input_tokens >= cfg.llambo_max_input_tokens:
            terminated_reason = "token_budget_exhausted"
            break
        if rounds >= max_rounds:
            terminated_reason = "max_rounds_safety_cap"
            break
        rounds += 1

        recent = history[-cfg.llambo_context_length:]
        history_text, normalized = _serialize_window(recent)
        best_normalized = max(normalized)

        n_uncond = -(-cfg.llambo_m // 2)  # ceil half
        n_cond = cfg.llambo_m // 2  # floor half
        target = min(100, round(best_normalized + cfg.llambo_alpha * 100))

        candidates: list[str] = []
        if n_uncond > 0:
            prompt = _build_prompt(tokenizer, _candidate_system_prompt(None), history_text)
            outs, in_len = _generate(model, tokenizer, prompt, n_uncond, cfg.llambo_temperature, cfg.llambo_top_p, MAX_NEW_TOKENS, device)
            total_input_tokens += in_len
            candidates.extend(c for c in outs if c)
        if n_cond > 0:
            prompt = _build_prompt(tokenizer, _candidate_system_prompt(target), history_text)
            outs, in_len = _generate(model, tokenizer, prompt, n_cond, cfg.llambo_temperature, cfg.llambo_top_p, MAX_NEW_TOKENS, device)
            total_input_tokens += in_len
            candidates.extend(c for c in outs if c)

        if not candidates:
            print(f"[{run_id} task {task_idx}] round {rounds}: all-empty generation, retrying")
            continue

        ranked: list[tuple[float, str]] = []
        for candidate in candidates:
            surrogate_prompt = _build_prompt(
                tokenizer, _surrogate_system_prompt(), f"{history_text}\n{candidate} -> "
            )
            outs, in_len = _generate(
                model, tokenizer, surrogate_prompt, cfg.llambo_k_mc_samples,
                cfg.llambo_temperature, cfg.llambo_top_p, SURROGATE_MAX_NEW_TOKENS, device,
            )
            total_input_tokens += in_len
            parsed = [p for p in (_parse_int_0_100(o) for o in outs) if p is not None]
            if parsed:
                mu = sum(parsed) / len(parsed) / 100.0
                sigma = (sum((p / 100.0 - mu) ** 2 for p in parsed) / len(parsed)) ** 0.5
            else:
                mu, sigma = 0.0, 0.0
            ei = _expected_improvement(mu, sigma, best_normalized / 100.0, cfg.llambo_alpha)
            ranked.append((ei, candidate))

        ranked.sort(key=lambda pair: pair[0], reverse=True)
        batch = [candidate for _ei, candidate in ranked[: cfg.llambo_batch_size]]
        batch = batch[: cfg.oracle_budget - n_extra_calls]
        if not batch:
            continue

        real_scores = list(-apex_wrapper(batch)[:, 0])
        history.extend(zip(batch, real_scores))
        n_extra_calls += len(batch)

    if terminated_reason != "oracle_budget_reached":
        print(
            f"[{run_id} task {task_idx}] terminated early after {rounds} rounds "
            f"({terminated_reason}): {n_extra_calls}/{cfg.oracle_budget} post-init oracle calls, "
            f"{total_input_tokens}/{cfg.llambo_max_input_tokens} input tokens"
        )

    with dest_csv.open("w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["train_x", "train_y"])
        writer.writerows(history)
    meta_path.write_text(json.dumps({
        "n_oracle_calls_completed": n_extra_calls,
        "total_input_tokens": total_input_tokens,
        "terminated_reason": terminated_reason,
    }, indent=2))
    return dest_csv
