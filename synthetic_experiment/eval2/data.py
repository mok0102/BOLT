"""Readers for synthetic Branin heldout trajectories."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def task_scores(run_dir: Path, arm: str, milestone: int, task_index: int) -> pd.Series | None:
    directory = arm if arm in {'STBO', 'LLAMBO'} else f'{arm}-{milestone}'
    path = run_dir / 'heldout' / directory / f'task_{task_index:04d}.csv'
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if 'train_y' not in frame or frame.empty:
        raise ValueError(f'{path}: expected nonempty train_y column')
    return frame['train_y'].astype(float).reset_index(drop=True)


def task_curves(run_dir: Path, arm: str, milestone: int, num_tasks: int, init_size: int, oracle_budget: int) -> pd.DataFrame:
    curves = {}
    for index in range(num_tasks):
        scores = task_scores(run_dir, arm, milestone, index)
        if scores is None or len(scores) < init_size:
            continue
        curve = scores.cummax().iloc[init_size - 1:init_size + oracle_budget].reset_index(drop=True)
        curves[index] = curve
    return pd.DataFrame(curves)


def init_curves(run_dir: Path, arm: str, milestone: int, num_tasks: int, init_size: int) -> pd.DataFrame:
    curves = {}
    for index in range(num_tasks):
        scores = task_scores(run_dir, arm, milestone, index)
        if scores is not None and len(scores) >= init_size:
            curves[index] = scores.iloc[:init_size].cummax().reset_index(drop=True)
    result = pd.DataFrame(curves)
    result.index += 1
    return result
