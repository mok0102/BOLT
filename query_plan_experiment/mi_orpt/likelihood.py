"""Reference likelihood of assistant plan tokens, conditioned on the real SQL."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "You are a helpful assistant that provides efficient orderings for given queries."

def load_reference(checkpoint, base):
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).cuda().eval()
    return model, tokenizer

def score_sequences(model, tokenizer, context, sequences):
    messages = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":context}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    scores = {}
    with torch.inference_mode():
        for seq in sequences:
            ids = tokenizer.apply_chat_template(messages+[{"role":"assistant","content":seq}], tokenize=True, return_dict=False)
            if ids[:len(prefix)] != prefix:
                raise ValueError("Reference chat prefix does not match assistant boundary")
            x = torch.tensor([ids], device=model.device)
            logits = model(x).logits[:, len(prefix)-1:-1].float()
            targets = x[:, len(prefix):]
            score = logits.log_softmax(-1).gather(-1,targets.unsqueeze(-1)).sum().item()
            scores[seq] = score
    return scores
