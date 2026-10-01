"""Shared candidate generation and dependency-light continuous GP-BO steps."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .branin import BRANIN_BOUNDS, BraninTask
from .config import ExperimentConfig
from .prompts import messages, parse_point

FINE_TUNING_DIR = "fine-tuning/query_plans"


def _run(cmd: list, cwd: Path, cfg: ExperimentConfig | None = None) -> None:
    print(f"+ ({cwd}) {' '.join(map(str, cmd))}", flush=True)
    env = {**os.environ, "PAGER": "cat", "MANPAGER": "cat", "GIT_PAGER": "cat"}
    if cfg and cfg.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True, env=env, stdin=subprocess.DEVNULL)


def cleanup_intermediate_epochs(root: Path, keep: Path) -> None:
    for path in root.glob("epoch_*"):
        if path.is_dir() and path != keep:
            shutil.rmtree(path)


def _uniform_points(rng: np.random.Generator, n: int) -> np.ndarray:
    lows = np.array([b[0] for b in BRANIN_BOUNDS])
    highs = np.array([b[1] for b in BRANIN_BOUNDS])
    return rng.uniform(lows, highs, size=(n, 2))


def _scaled(x: np.ndarray) -> np.ndarray:
    lows = np.array([b[0] for b in BRANIN_BOUNDS])
    spans = np.array([b[1] - b[0] for b in BRANIN_BOUNDS])
    return (x - lows) / spans


def _gp_ucb_candidate(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    pool_size: int,
    *,
    lengthscale: float = 0.2,
    beta: float = 2.0,
) -> np.ndarray:
    """Fit a tiny exact RBF GP and maximize UCB over a uniform candidate bank."""
    candidates = _uniform_points(rng, pool_size)
    sx, sc = _scaled(x), _scaled(candidates)
    kxx = np.exp(-0.5 * np.sum((sx[:, None] - sx[None, :]) ** 2, axis=-1) / lengthscale**2)
    kxc = np.exp(-0.5 * np.sum((sx[:, None] - sc[None, :]) ** 2, axis=-1) / lengthscale**2)
    centered = y - y.mean()
    scale = max(float(y.std()), 1.0)
    try:
        alpha = np.linalg.solve(kxx + 1e-6 * np.eye(len(x)), centered / scale)
        solved = np.linalg.solve(kxx + 1e-6 * np.eye(len(x)), kxc)
    except np.linalg.LinAlgError:
        return candidates[0]
    mean = y.mean() + scale * (kxc.T @ alpha)
    variance = np.maximum(1.0 - np.sum(kxc * solved, axis=0), 1e-12)
    return candidates[int(np.argmax(mean + beta * scale * np.sqrt(variance)))]


def _load_hf_model(checkpoint: Path, base: Path):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), tokenizer, device


def sample_from_checkpoint(
    cfg: ExperimentConfig,
    checkpoint: Path,
    task_t: float,
    n: int,
    seed: int,
    output_path: Path | None = None,
    loaded_model=None,
) -> np.ndarray:
    """Sample coordinates and persist raw/parsing/fallback provenance as JSONL."""
    import torch

    model, tokenizer, device = (
        loaded_model
        if loaded_model is not None
        else _load_hf_model(checkpoint, cfg.base_checkpoint_dir)
    )
    prompt = tokenizer.apply_chat_template(messages(task_t), tokenize=False, add_generation_prompt=True)
    # transformers versions differ on whether generate() accepts a
    # `generator` model kwarg. Seed torch directly for portable,
    # deterministic sampling instead of forwarding that version-specific
    # argument to generate().
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    found: list[list[float]] = []
    seen = set()
    records: list[dict] = []
    for attempt in range(1, 6):
        inputs = tokenizer([prompt] * max(n, 4), return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=True, temperature=cfg.sampling_temperature,
                                    max_new_tokens=32,
                                    pad_token_id=tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for sample_index, text in enumerate(decoded):
            point = parse_point(text)
            key = tuple(point) if point else None
            if point is None:
                accepted, reason = False, "parse_or_bounds_failure"
            elif key in seen:
                accepted, reason = False, "duplicate"
            elif len(found) >= n:
                accepted, reason = False, "quota_filled"
            else:
                accepted, reason = True, "accepted"
                seen.add(key)
                found.append(point)
            records.append({
                "source": "llm",
                "checkpoint": str(checkpoint),
                "task_t": task_t,
                "seed": seed,
                "attempt": attempt,
                "sample_index": sample_index,
                "prompt": prompt,
                "raw_output": text,
                "parsed_point": point,
                "accepted": accepted,
                "reason": reason,
            })
        if len(found) >= n:
            break
    rng = np.random.default_rng(seed)
    extra = _uniform_points(rng, n - len(found))
    for point in extra:
        records.append({
            "source": "random_fallback",
            "checkpoint": str(checkpoint),
            "task_t": task_t,
            "seed": seed,
            "attempt": None,
            "sample_index": None,
            "prompt": prompt,
            "raw_output": None,
            "parsed_point": point.tolist(),
            "accepted": True,
            "reason": "llm_valid_pool_shortfall",
        })
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(record) + "\n" for record in records))
        tmp.replace(output_path)
    return np.vstack([np.asarray(found).reshape(-1, 2), extra])


def initial_points(
    cfg: ExperimentConfig,
    task_t: float,
    seed: int,
    checkpoint: Path | None = None,
    sampling_output_path: Path | None = None,
    loaded_model=None,
) -> np.ndarray:
    if checkpoint is not None and cfg.proposal_source == "llm":
        return sample_from_checkpoint(
            cfg, checkpoint, task_t, cfg.init_size, seed, sampling_output_path, loaded_model
        )
    return _uniform_points(np.random.default_rng(seed), cfg.init_size)


def write_trajectory(path: Path, task_t: float, x: np.ndarray, y: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["train_x", "train_y", "task_t"])
        writer.writeheader()
        for point, score in zip(x, y):
            writer.writerow({"train_x": json.dumps(point.tolist()), "train_y": float(score), "task_t": task_t})
    tmp.replace(path)
    return path


def run_bo(cfg: ExperimentConfig, task_t: float, destination: Path, *, seed: int, checkpoint: Path | None = None,
           initial_x: np.ndarray | None = None, max_bo_steps: int | None = None) -> Path:
    if destination.exists():
        return destination
    rng = np.random.default_rng(seed)
    sampling_output_path = None
    if checkpoint is not None and cfg.proposal_source == "llm":
        sampling_output_path = globals()["sampling_output_path"](cfg, checkpoint, destination)
    x = np.asarray(
        initial_x if initial_x is not None else initial_points(
            cfg, task_t, seed, checkpoint, sampling_output_path
        ),
        dtype=float,
    )
    task = BraninTask(task_t)
    y = np.asarray(task(x), dtype=float)
    steps = cfg.oracle_budget if max_bo_steps is None else max_bo_steps
    partial = destination.with_suffix(".partial.csv")
    print(
        f"[BO] start task_t={task_t:.3f} init={len(x)} steps={steps} "
        f"checkpoint={checkpoint} destination={destination}",
        flush=True,
    )
    write_trajectory(partial, task_t, x, y)
    for step in range(steps):
        point = _gp_ucb_candidate(
            x,
            y,
            rng,
            cfg.bo_candidate_pool_size,
            lengthscale=cfg.bo_lengthscale,
            beta=cfg.bo_ucb_beta,
        )
        x = np.vstack([x, point])
        y = np.append(y, task(point))
        write_trajectory(partial, task_t, x, y)
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            print(
                f"[BO] task_t={task_t:.3f} step={step + 1}/{steps} "
                f"best={float(y.max()):.6f}",
                flush=True,
            )
    result = write_trajectory(destination, task_t, x, y)
    partial.unlink(missing_ok=True)
    return result


def sampling_output_path(cfg: ExperimentConfig, checkpoint: Path, destination: Path) -> Path:
    """Stable raw-generation log path shared by serial and batched callers."""

    checkpoint_label = next(
        (part for part in reversed(checkpoint.parts) if re.fullmatch(r"(?:BOLT|ORPT)-\d+", part)),
        "checkpoint",
    )
    try:
        relative = destination.relative_to(cfg.run_dir).with_suffix("")
        output_name = "__".join(relative.parts) + ".jsonl"
    except ValueError:
        output_name = destination.stem + ".jsonl"
    return cfg.run_dir / "sampling" / checkpoint_label / output_name


def checkpoint_milestone(path: Path) -> int:
    match = next((re.fullmatch(r"(?:BOLT|ORPT)-(\d+)", p) for p in reversed(path.parts)
                  if re.fullmatch(r"(?:BOLT|ORPT)-(\d+)", p)), None)
    if not match:
        raise ValueError(f"Cannot determine milestone from {path}")
    return int(match.group(1))
