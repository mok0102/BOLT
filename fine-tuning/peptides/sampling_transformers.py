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


def generate_answers(model, tokenizer, peptide: str, args, device: str) -> list[str]:
    prompt = build_prompt(tokenizer, peptide)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = encoded["input_ids"].shape[-1]

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.samples_per_peptide,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    answers = []
    for sequence in generated:
        new_tokens = sequence[input_length:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
        answers.append(clean_generation(answer))
    return answers


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


def main() -> None:
    args = parse_args()
    args.model_path = args.model_path.expanduser().resolve()
    if args.tokenizer_path is not None:
        args.tokenizer_path = args.tokenizer_path.expanduser().resolve()
    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {args.model_path}. "
            "Check that the previous fine-tuning run produced the expected epoch directory."
        )

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer_path = args.tokenizer_path or args.model_path

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using model: {args.model_path}")
    print(f"Using tokenizer: {tokenizer_path}")
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    print(f"Using temperature: {args.temperature}")
    if args.reference_sequence:
        print("Using explicit reference sequence")
    else:
        print(
            f"Using reference sequence slice: "
            f"[{args.start_index}:{args.start_index + args.num_peptides}]"
        )

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
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    source_peptides = get_source_peptides(args)
    print("Reference sequences to sample:")
    for peptide_index, peptide in enumerate(source_peptides, start=1):
        print(f"  {peptide_index}: {peptide}")

    with args.output_file.open("w") as f:
        for peptide in tqdm(source_peptides, desc="Generating variants"):
            answers = generate_answers(model, tokenizer, peptide, args, device)
            record = {
                "source_peptide": peptide,
                "generated_answers": answers,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved results to: {args.output_file}")


if __name__ == "__main__":
    main()
