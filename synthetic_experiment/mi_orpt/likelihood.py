"""Reference-policy likelihood q_t used to draw matched backgrounds."""

from __future__ import annotations

from ..prompts import messages
from ..steps import _load_hf_model


def score_sequences(cfg, checkpoint, task_t: float, sequences: list[str]) -> dict[str, float]:
    """Score assistant coordinate tokens; random proposals use uniform q_t."""
    if cfg.proposal_source == "random":
        return {seq: 0.0 for seq in sequences}
    import torch

    model, tokenizer, _ = _load_hf_model(checkpoint, cfg.base_checkpoint_dir)
    prompt_messages = messages(task_t)
    # Some transformers/tokenizers combinations return tokenizers.Encoding
    # from apply_chat_template(tokenize=True), which torch.tensor cannot
    # consume. Render text first and ask the tokenizer explicitly for a PT
    # tensor; this is stable across slow/fast tokenizer implementations.
    prefix_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prefix_ids = tokenizer(
        prefix_text, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(model.device)
    prefix_len = prefix_ids.shape[1]
    result = {}
    with torch.inference_mode():
        for seq in sequences:
            full_text = tokenizer.apply_chat_template(
                prompt_messages + [{"role": "assistant", "content": seq}],
                tokenize=False,
            )
            x = tokenizer(
                full_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device)
            if x.shape[1] <= prefix_len or not torch.equal(x[:, :prefix_len], prefix_ids):
                raise ValueError("Reference chat prefix does not match assistant boundary")
            logits = model(x).logits[:, prefix_len - 1:-1].float()
            targets = x[:, prefix_len:]
            result[seq] = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).sum().item()
    return result
