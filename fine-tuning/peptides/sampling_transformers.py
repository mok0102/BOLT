import argparse
import importlib.util
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "output" / "qwen_2_5_3B_output" / "epoch_4"
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "sampled_output_from_ft" / "sample_output_transformers.jsonl"
DEFAULT_REFERENCE_SEQUENCE = "RRYYEQLEQASRKGNRGFRR"

# Max sequences per model.generate() call. Peak GPU memory scales with this,
# not with the caller's requested sample count (see generate_for_reference).
GENERATE_CHUNK_SIZE = 250

SYSTEM_PROMPT = (
    "You are a specialized assistant that modifies peptide sequences to enhance "
    "antimicrobial activity. Make up to 25% sequence modifications based on known "
    "antimicrobial peptide properties such as: positive charge, hydrophobicity, "
    "and amphipathicity."
)

def load_reference_sequences() -> list[str]:
    refseqs_path = (
        SCRIPT_DIR / "../../optimization/peptides/apex_oracle/refseqs.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location("apex_refseqs", refseqs_path)
    if spec is None or spec.loader is None:
        return [DEFAULT_REFERENCE_SEQUENCE]

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(getattr(module, "REFERENCE_SEQUENCE", [DEFAULT_REFERENCE_SEQUENCE]))


try:
    REFERENCE_SEQUENCE = load_reference_sequences()
except Exception:
    REFERENCE_SEQUENCE = [DEFAULT_REFERENCE_SEQUENCE]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample peptide variants from a local Hugging Face transformers model."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index into apex_oracle.refseqs. Ignored when --reference-sequence is set.",
    )
    parser.add_argument("--num-peptides", type=int, default=1)
    parser.add_argument("--samples-per-peptide", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run generation on. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Torch dtype for model weights.",
    )
    parser.add_argument(
        "--reference-sequence",
        default=None,
        help="Use one explicit reference peptide instead of apex_oracle.refseqs.",
    )
    return parser.parse_args()


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


def build_prompt(tokenizer, peptide: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": peptide},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"{SYSTEM_PROMPT}\n\nUser: {peptide}\nAssistant:"


def clean_generation(text: str) -> str:
    text = text.strip()
    for marker in ("<|im_end|>", "<|endoftext|>", "</s>"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


def generate_for_reference(
    model,
    tokenizer,
    peptide: str,
    *,
    device: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    num_samples: int,
) -> list[str]:
    """Sample `num_samples` variants of one reference peptide.

    Split out of generate_answers() (kept below as a thin wrapper, so every
    existing caller is unaffected) purely so a long-lived worker can call it
    with an already-loaded model: see peptide_experiment/mi_orpt/
    warm_sampling_pool.py. Measured motivation -- invoking this module as a
    CLI once per task spends ~31s loading the checkpoint for ~7s of actual
    generation.
    """
    prompt = build_prompt(tokenizer, peptide)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = encoded["input_ids"].shape[-1]

    answers = []
    # Chunked rather than one num_return_sequences=num_samples call: peak GPU
    # memory scales with the batch, and mi_candidate_pool_size=2000 made a
    # single call reserve ~29GB. That fit (barely) when every call ran in a
    # fresh subprocess with a virgin allocator, but warm_sampling_pool.py's
    # workers are reused across hundreds of tasks and the caching allocator
    # fragments: a real OOM at milestone 400 reported 8.02GiB "reserved but
    # unallocated" alongside 25.89GiB live. Chunking bounds the peak for BOTH
    # the CLI and the pooled path (same total sample count either way --
    # sampling is unseeded, so nothing else changes).
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


def generate_answers(model, tokenizer, peptide: str, args, device: str) -> list[str]:
    return generate_for_reference(
        model,
        tokenizer,
        peptide,
        device=device,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.samples_per_peptide,
    )


def get_source_peptides(args) -> list[str]:
    if args.reference_sequence:
        return [args.reference_sequence]

    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if args.num_peptides < 1:
        raise ValueError("--num-peptides must be at least 1.")

    end_index = args.start_index + args.num_peptides
    source_peptides = list(REFERENCE_SEQUENCE[args.start_index : end_index])
    if not source_peptides:
        raise ValueError(
            f"No reference sequences found for start_index={args.start_index}, "
            f"num_peptides={args.num_peptides}. REFERENCE_SEQUENCE has "
            f"{len(REFERENCE_SEQUENCE)} entries."
        )
    return source_peptides


def load_model_and_tokenizer(model_path, tokenizer_path=None, device: str = "auto", dtype: str = "auto"):
    """Load the checkpoint + tokenizer once; returns (model, tokenizer, device).

    Extracted verbatim out of main() so that a persistent worker process can
    hold the result across many generate calls instead of paying this cost per
    task (peptide_experiment/mi_orpt/warm_sampling_pool.py). Keeps the
    AutoTokenizer -> PreTrainedTokenizerFast fallback and the pad-token fixup,
    both of which are load-bearing for this repo's torchtune-materialized
    checkpoints.
    """
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}. "
            "Check that the previous fine-tuning run produced the expected epoch directory."
        )
    tokenizer_path = Path(tokenizer_path).expanduser().resolve() if tokenizer_path is not None else model_path

    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)

    print(f"Using model: {model_path}")
    print(f"Using tokenizer: {tokenizer_path}")
    print(f"Using device: {resolved_device}")
    print(f"Using dtype: {resolved_dtype}")

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
            additional_special_tokens=[
                "<|im_start|>",
                "<|im_end|>",
            ],
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=resolved_dtype,
        trust_remote_code=True,
    )
    model.to(resolved_device)
    model.eval()
    return model, tokenizer, resolved_device


def write_records(output_file, records) -> None:
    """Single source of truth for this script's jsonl format.

    Both main() and warm_sampling_pool.py's worker go through here, so the CLI
    path and the pooled path cannot drift apart -- downstream
    sampled_output_from_ft/make_initialization_data.py parses these files and
    would silently mis-read a divergent schema.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path, args.tokenizer_path, args.device, args.dtype
    )

    print(f"Using temperature: {args.temperature}")
    if args.reference_sequence:
        print("Using explicit reference sequence")
    else:
        print(
            f"Using reference sequence slice: "
            f"[{args.start_index}:{args.start_index + args.num_peptides}]"
        )

    source_peptides = get_source_peptides(args)
    print("Reference sequences to sample:")
    for peptide_index, peptide in enumerate(source_peptides, start=1):
        print(f"  {peptide_index}: {peptide}")

    records = [
        {
            "source_peptide": peptide,
            "generated_answers": generate_answers(model, tokenizer, peptide, args, device),
        }
        for peptide in tqdm(source_peptides, desc="Generating variants")
    ]
    write_records(args.output_file, records)

    print(f"Saved results to: {args.output_file}")


if __name__ == "__main__":
    main()
