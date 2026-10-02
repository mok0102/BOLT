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

from .branin import BRANIN_BOUNDS
from .config import ExperimentConfig
from .prompts import messages, parse_point
from .task_splits import TaskRecord

FINE_TUNING_DIR = "fine-tuning/query_plans"


def tune_executable() -> str:
    """torchtune's `tune` CLI, resolved next to the running interpreter, never
    from PATH. This machine (see mol_experiment/ENVIRONMENT.md) has more than
    one venv that ships a `tune` binary with a different torch/torchtune
    build; a bare "tune" in argv resolves through PATH and can silently train
    through the wrong environment. sys.prefix, not Path(sys.executable).resolve()
    -- the latter follows a uv venv's symlink straight out of the venv."""
    import sys
    candidate = Path(sys.prefix) / "bin" / "tune"
    if not candidate.exists():
        raise RuntimeError(
            f"torchtune's `tune` CLI not found at {candidate} (sys.prefix={sys.prefix}). "
            "Launch synthetic_experiment with the interpreter of the venv that has it."
        )
    return str(candidate)


def torchrun_master_port(cfg: ExperimentConfig) -> int:
    """29500 + first visible GPU id, so concurrent trajectory_chain processes
    on different GPUs don't collide on torchrun's rendezvous port.

    Confirmed live (2026-10-02): two `tune run --nnodes 1 --nproc_per_node 1`
    launches started concurrently (one per trajectory_chain process, each
    pinned to its own GPU pair via CUDA_VISIBLE_DEVICES) both tried to bind
    torchrun's default rendezvous port 29500 and one died with
    `RuntimeError: ... EADDRINUSE`. The port is a host-wide TCP resource, not
    scoped per GPU, so CUDA_VISIBLE_DEVICES isolation alone does not prevent
    the collision. Mirrors peptide_experiment/mol_experiment's own
    `master_port = 29500 + int(cfg.mi_parallel_gpus[0])` fix for the identical
    failure mode there.
    """
    if not cfg.cuda_visible_devices:
        return 29500
    first_gpu = cfg.cuda_visible_devices.split(",")[0].strip()
    return 29500 + int(first_gpu)


def _run(cmd: list, cwd: Path, cfg: ExperimentConfig | None = None) -> None:
    print(f"+ ({cwd}) {' '.join(map(str, cmd))}", flush=True)
    env = {**os.environ, "PAGER": "cat", "MANPAGER": "cat", "GIT_PAGER": "cat"}
    if cfg and cfg.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True, env=env, stdin=subprocess.DEVNULL)


def materialize_hf_checkpoint(ckpt_dir: Path, epoch: int, base_checkpoint_dir: Path) -> Path:
    """Reconcile torchtune's actual LoRA-SFT/DPO checkpoint output into a
    loadable HF-format directory (``model.safetensors.index.json`` +
    tokenizer files), regardless of which on-disk layout this torchtune
    version used.

    Added 2026-10-02 after a reproducible, deterministic crash: torchtune
    0.4.0's FullModelHFCheckpointer logged "saved successfully" and wrote
    flat, epoch-suffixed files (``hf_model_000N_{epoch}.pt``,
    ``adapter_{epoch}.pt``) directly in ``output_dir`` -- not the nested
    ``epoch_{N}/`` subdirectory this module's training functions checked
    for. That happened identically across 4 independent concurrent
    `tune run` invocations (same epoch, same files, same traceback),
    ruling out a race condition -- it is a structural mismatch between this
    torchtune install's real output convention and the path this package
    assumed, not a flake.

    Byte-identical port of peptide_experiment/steps.py::materialize_hf_checkpoint
    (via mol_experiment's own byte-identical port, mol_experiment/
    ENVIRONMENT.md) -- that domain hit and fixed the identical torchtune
    0.4.0-vs-newer layout difference first; ported here rather than
    re-derived, per the "reuse, don't reimplement" instruction in
    impl_plan/motivational_exp.txt.
    """
    import torch

    index_path = ckpt_dir / "model.safetensors.index.json"
    if index_path.exists():
        return ckpt_dir

    epoch_dir = ckpt_dir / f"epoch_{epoch}"
    if (epoch_dir / "model.safetensors.index.json").exists():
        for item in list(epoch_dir.iterdir()):
            is_stray_epoch_dir = item.is_dir() and item.name.startswith("epoch_") and item.name[len("epoch_"):].isdigit()
            if item.is_dir() and (item.name == "recipe_state" or is_stray_epoch_dir):
                shutil.rmtree(item)
                continue
            shutil.move(str(item), str(ckpt_dir / item.name))
        epoch_dir.rmdir()
        return ckpt_dir

    from safetensors.torch import save_file

    shards = sorted(ckpt_dir.glob(f"hf_model_*_{epoch}.pt"))
    if not shards:
        raise FileNotFoundError(
            f"materialize_hf_checkpoint: no hf_model_*_{epoch}.pt shards in {ckpt_dir} "
            "(expected torchtune's FullModelHFCheckpointer to have written them)"
        )
    num_shards = len(shards)
    for cpt_idx, shard_path in enumerate(shards, start=1):
        state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
        seen_ptrs: dict[int, str] = {}
        for key, tensor in state_dict.items():
            ptr = tensor.data_ptr()
            if ptr in seen_ptrs:
                state_dict[key] = tensor.clone()
            else:
                seen_ptrs[ptr] = key
        out_path = ckpt_dir / f"model-{cpt_idx:05d}-of-{num_shards:05d}.safetensors"
        save_file(state_dict, out_path, metadata={"format": "pt"})

    base_index = base_checkpoint_dir / "model.safetensors.index.json"
    if not base_index.exists():
        raise FileNotFoundError(
            f"materialize_hf_checkpoint: base checkpoint has no model.safetensors.index.json at {base_index}"
        )
    shutil.copy2(base_index, index_path)

    for tokenizer_file in ("vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"):
        src = base_checkpoint_dir / tokenizer_file
        dst = ckpt_dir / tokenizer_file
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    return ckpt_dir


def checkpoint_ready(ckpt_dir: Path) -> bool:
    """True once ``ckpt_dir`` holds either a materialized HF checkpoint or a
    ``proposal_source: random`` dummy-checkpoint marker. A bare
    ``ckpt_dir.exists()`` is NOT a valid completion check: torchtune (or
    ``_run``'s own subprocess) creates ``output_dir`` well before training
    finishes, so an early-return keyed on mere directory existence would
    treat an in-progress or previously-interrupted run as already done."""
    return (ckpt_dir / "model.safetensors.index.json").exists() or (ckpt_dir / "RANDOM_PROPOSAL_MARKER").exists()


def cleanup_intermediate_epochs(ckpt_dir: Path, final_epoch: int) -> None:
    """Byte-identical port of peptide_experiment/steps.py::cleanup_intermediate_epochs
    (via mol_experiment). Handles both this torchtune install's flat
    ``hf_model_*_*.pt``/``adapter_*.pt`` files and the nested ``epoch_*/``
    layout, deleting every non-final-epoch artifact once the checkpoint is
    materialized -- replaces this module's own prior ``cleanup_intermediate_
    epochs(root, keep: Path)``, which only ever globbed for ``epoch_*``
    directories and so never cleaned up the flat-file layout this install
    actually produces."""
    materialized = (ckpt_dir / "model.safetensors.index.json").exists()
    for pattern in ("hf_model_*_*.pt", "adapter_*.pt"):
        for f in ckpt_dir.glob(pattern):
            try:
                epoch = int(f.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if epoch != final_epoch or materialized:
                f.unlink()
    recipe_state = ckpt_dir / "recipe_state.pt"
    if recipe_state.exists():
        recipe_state.unlink()
    for d in ckpt_dir.glob("epoch_*"):
        if not d.is_dir():
            continue
        suffix = d.name[len("epoch_"):]
        if not suffix.isdigit():
            continue
        epoch = int(suffix)
        if epoch != final_epoch or materialized:
            shutil.rmtree(d)
    recipe_state_dir = ckpt_dir / "recipe_state"
    if recipe_state_dir.is_dir():
        shutil.rmtree(recipe_state_dir)


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
    task: TaskRecord,
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
    prompt = tokenizer.apply_chat_template(messages(task.transform), tokenize=False, add_generation_prompt=True)
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
                "task_split": task.split,
                "task_index": task.index,
                "task_descriptor": task.descriptor,
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
            "task_split": task.split,
            "task_index": task.index,
            "task_descriptor": task.descriptor,
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
    task: TaskRecord,
    seed: int,
    checkpoint: Path | None = None,
    sampling_output_path: Path | None = None,
    loaded_model=None,
) -> np.ndarray:
    if checkpoint is not None and cfg.proposal_source == "llm":
        return sample_from_checkpoint(
            cfg, checkpoint, task, cfg.init_size, seed, sampling_output_path, loaded_model
        )
    return _uniform_points(np.random.default_rng(seed), cfg.init_size)


def write_trajectory(path: Path, cfg: ExperimentConfig, task: TaskRecord, x: np.ndarray, y: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fieldnames = ["train_x", "train_y", "task_split", "task_index", "task_manifest"]
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point, score in zip(x, y):
            writer.writerow({
                "train_x": json.dumps(point.tolist()), "train_y": float(score),
                "task_split": task.split, "task_index": task.index, "task_manifest": cfg.manifest.token,
            })
    tmp.replace(path)
    return path


def run_bo(cfg: ExperimentConfig, task: TaskRecord, destination: Path, *, seed: int, checkpoint: Path | None = None,
           initial_x: np.ndarray | None = None, max_bo_steps: int | None = None) -> Path:
    if destination.exists():
        return destination
    rng = np.random.default_rng(seed)
    sampling_output_path = None
    if checkpoint is not None and cfg.proposal_source == "llm":
        sampling_output_path = globals()["sampling_output_path"](cfg, checkpoint, destination)
    x = np.asarray(
        initial_x if initial_x is not None else initial_points(
            cfg, task, seed, checkpoint, sampling_output_path
        ),
        dtype=float,
    )
    oracle = task.oracle()
    y = np.asarray(oracle(x), dtype=float)
    steps = cfg.oracle_budget if max_bo_steps is None else max_bo_steps
    partial = destination.with_suffix(".partial.csv")
    print(
        f"[BO] start task={task.split}/{task.name} init={len(x)} steps={steps} "
        f"checkpoint={checkpoint} destination={destination}",
        flush=True,
    )
    write_trajectory(partial, cfg, task, x, y)
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
        y = np.append(y, oracle(point))
        write_trajectory(partial, cfg, task, x, y)
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            print(
                f"[BO] task={task.split}/{task.name} step={step + 1}/{steps} "
                f"best={float(y.max()):.6f}",
                flush=True,
            )
    result = write_trajectory(destination, cfg, task, x, y)
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
