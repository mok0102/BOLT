"""LLM candidate sampling for the mol domain. Mirrors
fine-tuning/peptides/sampling_transformers.py's shape (read-only reference,
never imported -- isolation contract): HF transformers AutoModelForCausalLM
generation from an already-fine-tuned (or base) Qwen checkpoint, chat-
template prompting, chunked model.generate() calls to bound peak GPU memory
regardless of the caller's requested sample count.

Deliberately separate from mi_orpt/likelihood.py's torchtune-based loader:
that one is for reference-model log-likelihood SCORING only (no .generate()
support in this repo's torchtune usage) and is a different concern.

Differs from sampling_transformers.py in what a "valid candidate" means:
peptide's raw amino-acid completions need no parsing (the raw string IS the
candidate). mol's completions must be, in order:
  1. parsed (mol_prompt.parse_candidate_selfies -- rejects anything that
     isn't a bare SELFIES-bracket string: prose, code fences, multiple
     molecules);
  2. vocab-checked against the LOL-BO SELFIES-VAE's fixed vocabulary (same
     requirement as everywhere else a SELFIES string feeds the VAE --
     mol_init_candidates.py, fine-tuning/mol/make_train_data_csv.py,
     mol_experiment/mi_orpt/build_pairs.py -- an out-of-vocab token causes a
     KeyError deep inside SELFIESDataset's encoder, see
     MOL_LATENT_SPACE_FINDING.md's documented fix for the same class of bug
     in mol_init_candidates.py);
  3. decoded + RDKit-validated + canonicalized (mol_oracle.py's own
     decode_and_canonicalize, reused here rather than re-implemented, so the
     empty-SMILES trap documented there is guarded exactly once).
selfies_to_canonical_smiles_vocab_checked below is the one function that
does all three and returns the canonical-SMILES representation every other
mol module already uses internally (candidate_bank.py, mol_run_bo's
trajectory CSVs, ...).
"""

from __future__ import annotations

import sys
from pathlib import Path

import selfies as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mol_oracle import decode_and_canonicalize  # noqa: E402
from mol_prompt import SYSTEM_PROMPT, render_context  # noqa: E402

# Max sequences per model.generate() call -- peak GPU memory scales with
# this, not with the caller's requested sample count (mirrors
# sampling_transformers.py's own GENERATE_CHUNK_SIZE rationale).
GENERATE_CHUNK_SIZE = 250

# SELFIES strings for this task's manifest range up to 858 characters
# (measured: optimization/mol/task_manifest/mol_task_manifest.tsv's
# seed_selfies column, min/median/max = 44/216/858) -- 512 leaves headroom
# for a BPE tokenizer without bracket-aware merges to still fit the longest
# real candidates.
DEFAULT_MAX_NEW_TOKENS = 512

_VOCAB = None


def _vocab() -> dict:
    global _VOCAB
    if _VOCAB is None:
        from mol_lolbo.utils.mol_utils.selfies_vae.model_positional_unbounded import SELFIESDataset

        _VOCAB = SELFIESDataset().vocab2idx
    return _VOCAB


def selfies_to_canonical_smiles_vocab_checked(selfies_str: str) -> str | None:
    """Accepts a raw candidate SELFIES string only if it's syntactically
    splittable, decodes to a valid, non-empty molecule, AND -- the check that
    actually matters -- every token of the SELFIES obtained by RE-ENCODING
    that molecule's canonical SMILES (sf.encoder(canon), not the original
    selfies_str) is in the VAE's fixed vocabulary. Returns canonical SMILES,
    or None if any check fails.

    Checking the *original* LLM-output tokens (an earlier version of this
    function did only that) is not sufficient: MoleculeObjective.vae_forward
    (mol_objective.py) re-derives SELFIES from the canonical SMILES via
    sf.encoder(smile) rather than reusing the LLM's raw string, and SELFIES
    re-encoding of a canonicalized molecule can introduce tokens the original
    string never had. Confirmed in production (2026-09-30): a task's LLM
    output passed the original-tokens check, but the canonical-SMILES
    round-trip re-encoding produced '[=Se-1]', not in vocab and never
    checked, which then crashed SELFIESDataset's raw dict-indexed encode()
    deep inside vae_forward with an unhandled KeyError. mol_init_candidates.py
    already checks the re-encoded form correctly (see its own generate-and-
    filter loop) -- this mirrors that, not a new pattern."""
    try:
        tokens = list(sf.split_selfies(selfies_str))
    except Exception:
        return None
    if not tokens:
        return None
    canon = decode_and_canonicalize(selfies_str, input_kind="selfies")
    if canon is None:
        return None
    try:
        re_encoded_tokens = list(sf.split_selfies(sf.encoder(canon)))
    except Exception:
        return None
    vocab = _vocab()
    if not all(t in vocab for t in re_encoded_tokens):
        return None
    return canon


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device_arg


def resolve_dtype(dtype_arg: str, device: str):
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(model_path, tokenizer_path=None, device: str = "auto", dtype: str = "auto"):
    """Load the checkpoint + tokenizer once for generation. Keeps the
    AutoTokenizer -> PreTrainedTokenizerFast fallback and the pad-token
    fixup from sampling_transformers.py's own loader, both load-bearing for
    this repo's torchtune-materialized checkpoints."""
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}. "
            "Check that the previous fine-tuning run produced the expected checkpoint directory."
        )
    tokenizer_path = Path(tokenizer_path).expanduser().resolve() if tokenizer_path is not None else model_path

    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)

    print(f"[mol_sampling] model: {model_path}, tokenizer: {tokenizer_path}, device: {resolved_device}, dtype: {resolved_dtype}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except ValueError:
        tokenizer_json = Path(tokenizer_path) / "tokenizer.json"
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

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=resolved_dtype, trust_remote_code=True)
    model.to(resolved_device)
    model.eval()
    return model, tokenizer, resolved_device


def build_prompt(tokenizer, target_sequence: str, seed_selfies: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_context(target_sequence, seed_selfies)},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\nUser: {render_context(target_sequence, seed_selfies)}\nAssistant:"


def clean_generation(text: str) -> str:
    text = text.strip()
    for marker in ("<|im_end|>", "<|endoftext|>", "</s>"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


def generate_for_task(
    model,
    tokenizer,
    target_sequence: str,
    seed_selfies: str,
    *,
    device: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    num_samples: int,
) -> list[str]:
    """Raw (unparsed, unvalidated) completions -- caller runs
    mol_prompt.parse_candidate_selfies + selfies_to_canonical_smiles_vocab_checked
    on each before treating it as a usable candidate."""
    prompt = build_prompt(tokenizer, target_sequence, seed_selfies)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = encoded["input_ids"].shape[-1]

    answers = []
    for start in range(0, num_samples, GENERATE_CHUNK_SIZE):
        chunk = min(GENERATE_CHUNK_SIZE, num_samples - start)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=chunk,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for sequence in generated:
            new_tokens = sequence[input_length:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
            answers.append(clean_generation(answer))
        del generated
    return answers
