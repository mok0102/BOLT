"""Zero-shot LLAMBO-style candidate generation and in-context EI ranking."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from .branin import BraninTask
from .config import ExperimentConfig
from .prompts import format_task_t, parse_point, point_text
from .steps import _load_hf_model, _uniform_points, write_trajectory

_NUMBER = re.compile(r'(?<![\d.])-?\d+(?:\.\d+)?')


def _ei(mu: float, sigma: float, best: float, xi: float) -> float:
    margin = mu - best - xi
    if sigma <= 1e-9:
        return max(margin, 0.0)
    z = margin / sigma
    return margin * 0.5 * (1 + math.erf(z / math.sqrt(2))) + sigma * math.exp(-0.5 * z*z) / math.sqrt(2 * math.pi)


def _history(x: np.ndarray, y: np.ndarray, limit: int) -> tuple[str, float]:
    recent_x, recent_y = x[-limit:], y[-limit:]
    lo, hi = float(recent_y.min()), float(recent_y.max())
    normalized = np.full(len(recent_y), 50.0) if hi - lo < 1e-9 else 100 * (recent_y - lo) / (hi - lo)
    text = '\n'.join(f'{point_text(point)} -> {int(round(score))}' for point, score in zip(recent_x, normalized))
    return text, float(normalized.max()) / 100


def _generate(model, tokenizer, device, system: str, user: str, n: int, max_tokens: int) -> tuple[list[str], int]:
    import torch
    prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': system},
                                            {'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_length = inputs.input_ids.shape[1]
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.95,
                                num_return_sequences=n, max_new_tokens=max_tokens,
                                pad_token_id=tokenizer.pad_token_id)
    texts = tokenizer.batch_decode(output[:, input_length:], skip_special_tokens=True)
    return texts, input_length


def run_llambo_bo(cfg: ExperimentConfig, task_t: float, destination: Path, *, seed: int) -> Path:
    if destination.exists():
        return destination
    model, tokenizer, device = _load_hf_model(cfg.base_checkpoint_dir, cfg.base_checkpoint_dir)
    import torch
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x = _uniform_points(rng, cfg.init_size)
    task = BraninTask(task_t)
    y = np.asarray(task(x), dtype=float)
    partial = destination.with_suffix('.partial.csv')
    write_trajectory(partial, task_t, x, y)
    input_tokens = 0
    rounds = 0
    reason = 'oracle_budget_reached'
    while len(x) - cfg.init_size < cfg.oracle_budget:
        if input_tokens >= cfg.llambo_max_input_tokens:
            reason = 'token_budget_exhausted'
            break
        if rounds >= 2 * cfg.oracle_budget:
            reason = 'max_rounds_reached'
            break
        rounds += 1
        history, best = _history(x, y, cfg.llambo_context_length)
        user = f'task_t={format_task_t(task_t)}\nPrevious points and normalized scores (0-100):\n{history}'
        candidates = []
        counts = ((None, (cfg.llambo_candidates + 1)//2),
                  (min(100, round(100 + 100*cfg.llambo_alpha)), cfg.llambo_candidates//2))
        for target, count in counts:
            if count == 0:
                continue
            instruction = ('Propose a new Branin point expected to beat the best shown.' if target is None else
                           f'Propose a new Branin point expected to achieve normalized score {target}.')
            system = ('You optimize a two-dimensional Branin function. x1 must be in [-5,10], '
                      'x2 in [0,15]. ' + instruction + ' Return only [x1, x2].')
            texts, cost = _generate(model, tokenizer, device, system, user, count, 32)
            input_tokens += cost
            candidates.extend(point for text in texts if (point := parse_point(text)) is not None)
        if not candidates:
            continue
        ranked = []
        for point in candidates:
            system = ('Predict the normalized Branin score (0-100) for the candidate from the '
                      'observed points. Return only one number between 0 and 100.')
            texts, cost = _generate(model, tokenizer, device, system,
                                    f'{user}\nCandidate: {point_text(point)}\nPredicted score:',
                                    cfg.llambo_mc_samples, 8)
            input_tokens += cost
            predictions = []
            for text in texts:
                match = _NUMBER.search(text)
                if match:
                    predictions.append(max(0.0, min(100.0, float(match.group()))) / 100)
            if predictions:
                ranked.append((_ei(float(np.mean(predictions)), float(np.std(predictions)), best,
                                   cfg.llambo_alpha), point))
        if not ranked:
            continue
        point = max(ranked, key=lambda pair: pair[0])[1]
        x = np.vstack([x, point])
        y = np.append(y, task(point))
        write_trajectory(partial, task_t, x, y)
    result = write_trajectory(destination, task_t, x, y)
    partial.unlink(missing_ok=True)
    destination.with_suffix('.meta.json').write_text(json.dumps({'oracle_calls': len(x) - cfg.init_size,
        'input_tokens': input_tokens, 'terminated_reason': reason}, indent=2))
    return result
