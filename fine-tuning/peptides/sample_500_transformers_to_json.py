import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

import sampling_transformers as sampler


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "output" / "debug" / "sample_500_transformers.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample 500 peptide variants with sampling_transformers and save as JSON."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--reference-sequence", default=None)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    return parser.parse_args()


def load_tokenizer(tokenizer_path: Path):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except ValueError:
        tokenizer_json = tokenizer_path / "tokenizer.json"
        if not tokenizer_json.exists():
            raise
        return PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_json),
            eos_token="<|im_end|>",
            pad_token="<|endoftext|>",
            additional_special_tokens=[
                "<|im_start|>",
                "<|im_end|>",
            ],
        )


def get_reference_sequence(args: argparse.Namespace) -> str:
    if args.reference_sequence:
        return args.reference_sequence
    if args.start_index < 0 or args.start_index >= len(sampler.REFERENCE_SEQUENCE):
        raise ValueError(
            f"--start-index={args.start_index} is out of range for "
            f"{len(sampler.REFERENCE_SEQUENCE)} reference sequences."
        )
    return sampler.REFERENCE_SEQUENCE[args.start_index]


def main() -> None:
    args = parse_args()
    args.model_path = args.model_path.expanduser().resolve()
    if args.tokenizer_path is not None:
        args.tokenizer_path = args.tokenizer_path.expanduser().resolve()
    args.output_file = args.output_file.expanduser().resolve()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    device = sampler.resolve_device(args.device)
    dtype = sampler.resolve_dtype(args.dtype, device)
    tokenizer_path = args.tokenizer_path or args.model_path
    reference_sequence = get_reference_sequence(args)

    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    generation_args = argparse.Namespace(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        samples_per_peptide=args.num_samples,
    )
    generated_answers = sampler.generate_answers(
        model=model,
        tokenizer=tokenizer,
        peptide=reference_sequence,
        args=generation_args,
        device=device,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model_path": str(args.model_path),
        "tokenizer_path": str(tokenizer_path),
        "reference_sequence": reference_sequence,
        "start_index": None if args.reference_sequence else args.start_index,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "generated_answers": generated_answers,
    }
    with args.output_file.open("w") as f_out:
        json.dump(record, f_out, indent=2)
        f_out.write("\n")

    print(f"Reference sequence: {reference_sequence}")
    print(f"Saved {len(generated_answers)} samples to: {args.output_file}")


if __name__ == "__main__":
    main()
