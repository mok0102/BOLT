"""History-conditioned OptFormer training and heldout optimization for Branin."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .branin import BraninTaskTransform
from .config import ExperimentConfig
from .prompts import SYSTEM_PROMPT, parse_point, point_text, task_descriptor
from .steps import (
    FINE_TUNING_DIR, _load_hf_model, _run, _uniform_points, checkpoint_ready,
    cleanup_intermediate_epochs, materialize_hf_checkpoint, torchrun_master_port,
    tune_executable, write_trajectory,
)
from .task_splits import TaskRecord


def _score_bin(score: float, edges: list[float]) -> int:
    return int(np.searchsorted(edges, score, side='right'))


def _history_text(transform: BraninTaskTransform, x: np.ndarray, y: np.ndarray, edges: list[float], limit: int) -> str:
    rows = [f"{point_text(point)} -> bin {_score_bin(float(score), edges)}"
            for point, score in zip(x[-limit:], y[-limit:])]
    return f"{task_descriptor(transform)}\nObserved points (higher bin is better):\n" + "\n".join(rows) + "\nPropose the next point as [x1, x2]."


def train_optformer(cfg: ExperimentConfig, milestone: int) -> Path:
    if milestone not in cfg.milestones:
        raise ValueError(f"unknown milestone {milestone}")
    final = cfg.optformer_checkpoint_dir(milestone)
    if checkpoint_ready(final):
        return final
    tasks = cfg.train_tasks[:milestone]
    source_frames = []
    for task in tasks:
        path = cfg.trajectories_dir / f"{task.name}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        source_frames.append(pd.read_csv(path))
    all_scores = np.concatenate([frame.train_y.to_numpy(dtype=float) for frame in source_frames])
    edges = np.quantile(all_scores, np.linspace(0, 1, 101)[1:-1]).tolist()
    data_dir = cfg.run_dir / 'optformer'
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f'bin_edges_{milestone}.json').write_text(json.dumps(edges))
    rows = []
    for task, frame in zip(tasks, source_frames):
        x = np.asarray([json.loads(value) for value in frame.train_x], dtype=float)
        y = frame.train_y.to_numpy(dtype=float)
        eligible = np.arange(1, len(x))
        if len(eligible) > cfg.optformer_windows_per_task:
            eligible = np.unique(np.linspace(1, len(x) - 1, cfg.optformer_windows_per_task, dtype=int))
        for step in eligible:
            rows.append({'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': _history_text(task.transform, x[:step], y[:step], edges, cfg.optformer_context_length)},
                {'role': 'assistant', 'content': point_text(x[step])},
            ]})
    data_path = data_dir / f'train_data_{milestone}.jsonl'
    data_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    if cfg.proposal_source == 'random':
        raise ValueError('OptFormer requires proposal_source: llm and a base model checkpoint')
    from .orpt import check_training_size
    directory = cfg.bolt_root / FINE_TUNING_DIR
    check_training_size(data_path, directory / 'torchtune_config' / cfg.torchtune_config, cfg.optformer_epochs)
    out = cfg.checkpoints_dir / f'OptFormer-{milestone}'
    _run([tune_executable(), 'run', '--nnodes', '1', '--nproc_per_node', '1',
          '--master-port', str(torchrun_master_port(cfg)), cfg.torchtune_recipe,
          '--config', f'torchtune_config/{cfg.torchtune_config}', f'output_dir={out}',
          f'dataset.data_files={data_path}', f'checkpointer.checkpoint_dir={cfg.base_checkpoint_dir}',
          f'tokenizer.path={cfg.base_checkpoint_dir}/vocab.json',
          f'tokenizer.merges_file={cfg.base_checkpoint_dir}/merges.txt',
          f'epochs={cfg.optformer_epochs}', 'seed=42',
          f'metric_logger.log_dir={cfg.tensorboard_dir / f"OptFormer-{milestone}"}'], directory, cfg)
    materialize_hf_checkpoint(out, cfg.optformer_epochs - 1, cfg.base_checkpoint_dir)
    if not checkpoint_ready(final):
        raise RuntimeError(f'OptFormer checkpoint not produced: {final}')
    cleanup_intermediate_epochs(out, cfg.optformer_epochs - 1)
    return final


def run_optformer_bo(cfg: ExperimentConfig, task: TaskRecord, destination: Path, *, seed: int, milestone: int) -> Path:
    if destination.exists():
        return destination
    checkpoint = cfg.optformer_checkpoint_dir(milestone)
    if not checkpoint.exists():
        raise FileNotFoundError(f'Train OptFormer first: {checkpoint}')
    edges_path = cfg.run_dir / 'optformer' / f'bin_edges_{milestone}.json'
    edges = json.loads(edges_path.read_text())
    model, tokenizer, device = _load_hf_model(checkpoint, cfg.base_checkpoint_dir)
    import torch
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x = _uniform_points(rng, cfg.init_size)
    oracle = task.oracle()
    y = np.asarray(oracle(x), dtype=float)
    partial = destination.with_suffix('.partial.csv')
    write_trajectory(partial, cfg, task, x, y)
    fallback_count = 0
    for _ in range(cfg.oracle_budget):
        user = _history_text(task.transform, x, y, edges, cfg.optformer_context_length)
        prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': SYSTEM_PROMPT},
                                                {'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=True, temperature=cfg.sampling_temperature,
                                    max_new_tokens=32, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        point = parse_point(text)
        if point is None:
            point = _uniform_points(rng, 1)[0]
            fallback_count += 1
        x = np.vstack([x, point])
        y = np.append(y, oracle(point))
        write_trajectory(partial, cfg, task, x, y)
    result = write_trajectory(destination, cfg, task, x, y)
    partial.unlink(missing_ok=True)
    destination.with_suffix('.meta.json').write_text(json.dumps({'invalid_generation_fallbacks': fallback_count,
                                                                  'oracle_calls': cfg.oracle_budget}))
    return result
