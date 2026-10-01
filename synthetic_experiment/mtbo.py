"""Shared GP surrogate fitted across completed synthetic training tasks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .branin import BraninTask
from .config import ExperimentConfig
from .steps import _scaled, _uniform_points, write_trajectory
from .task_splits import task_name


def _features(x: np.ndarray, task_t: float) -> np.ndarray:
    return np.column_stack((_scaled(np.asarray(x, dtype=float)), np.full(len(x), task_t)))


def _kernel(a: np.ndarray, b: np.ndarray, lengthscale: float) -> np.ndarray:
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    return np.exp(-0.5 * d2 / lengthscale**2)


def train_mtbo_surrogate(cfg: ExperimentConfig, milestone: int) -> Path:
    """Pool top observations from the first ``milestone`` completed tasks."""
    if milestone not in cfg.milestones:
        raise ValueError(f"unknown milestone {milestone}")
    destination = cfg.mtbo_checkpoint(milestone)
    if destination.exists():
        with np.load(destination) as saved:
            if (int(saved['milestone']) == milestone and
                int(saved['top_n']) == cfg.mtbo_top_n_per_task and
                int(saved['max_points']) == cfg.mtbo_max_train_points and
                float(saved['lengthscale']) == cfg.bo_lengthscale):
                return destination
    chunks_x, chunks_y = [], []
    for index in range(milestone):
        source = cfg.trajectories_dir / f"{task_name(index)}.csv"
        if not source.exists():
            raise FileNotFoundError(source)
        frame = pd.read_csv(source).nlargest(cfg.mtbo_top_n_per_task, 'train_y')
        if frame.empty:
            raise ValueError(f"{source}: no observations")
        x = np.asarray([json.loads(value) for value in frame.train_x], dtype=float)
        chunks_x.append(_features(x, cfg.train_task_values[index]))
        chunks_y.append(frame.train_y.to_numpy(dtype=float))
    x, y = np.concatenate(chunks_x), np.concatenate(chunks_y)
    if len(x) > cfg.mtbo_max_train_points:
        # Fixed seed and ordered task chunks make the pooled subset reproducible.
        selected = np.sort(np.random.default_rng(cfg.bo_seed).choice(
            len(x), cfg.mtbo_max_train_points, replace=False))
        x, y = x[selected], y[selected]
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, x=x, y=y, milestone=milestone,
             top_n=cfg.mtbo_top_n_per_task, max_points=cfg.mtbo_max_train_points,
             lengthscale=cfg.bo_lengthscale)
    return destination


class _SharedGP:
    def __init__(self, path: Path, cfg: ExperimentConfig):
        with np.load(path) as saved:
            if (float(saved['lengthscale']) != cfg.bo_lengthscale or
                int(saved['top_n']) != cfg.mtbo_top_n_per_task or
                int(saved['max_points']) != cfg.mtbo_max_train_points):
                raise ValueError(f"{path}: MTBO settings differ from checkpoint")
            self.x, y = saved['x'], saved['y']
        self.lengthscale = cfg.bo_lengthscale
        self.center = float(y.mean())
        self.scale = max(float(y.std()), 1.0)
        k = _kernel(self.x, self.x, self.lengthscale)
        self.chol = np.linalg.cholesky(k + 1e-4 * np.eye(len(self.x)))
        self.alpha = np.linalg.solve(self.chol.T, np.linalg.solve(self.chol, (y - self.center) / self.scale))

    def mean(self, features: np.ndarray) -> np.ndarray:
        return self.center + self.scale * (_kernel(self.x, features, self.lengthscale).T @ self.alpha)


def run_mtbo_bo(cfg: ExperimentConfig, task_t: float, destination: Path, *, seed: int, milestone: int) -> Path:
    if destination.exists():
        return destination
    checkpoint = cfg.mtbo_checkpoint(milestone)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Train MTBO first: {checkpoint}")
    shared = _SharedGP(checkpoint, cfg)
    rng = np.random.default_rng(seed)
    x = _uniform_points(rng, cfg.init_size)
    task = BraninTask(task_t)
    y = np.asarray(task(x), dtype=float)
    partial = destination.with_suffix('.partial.csv')
    write_trajectory(partial, task_t, x, y)
    for _ in range(cfg.oracle_budget):
        candidates = _uniform_points(rng, cfg.bo_candidate_pool_size)
        observed = _features(x, task_t)
        proposed = _features(candidates, task_t)
        # Adapt the shared prediction to observed target-task residuals.
        residual = (y - shared.mean(observed)) / shared.scale
        kxx = _kernel(observed, observed, shared.lengthscale) + 1e-5 * np.eye(len(x))
        kxc = _kernel(observed, proposed, shared.lengthscale)
        chol = np.linalg.cholesky(kxx)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, residual))
        solved = np.linalg.solve(chol, kxc)
        mean = shared.mean(proposed) + shared.scale * (kxc.T @ alpha)
        variance = shared.scale**2 * np.maximum(1.0 - np.sum(solved**2, axis=0), 1e-8)
        acquisition = mean + cfg.bo_ucb_beta * np.sqrt(variance)
        distance = np.linalg.norm(_scaled(candidates)[:, None] - _scaled(x)[None, :], axis=-1)
        acquisition[np.min(distance, axis=1) < 1e-6] = -np.inf
        point = candidates[int(np.argmax(acquisition))]
        x = np.vstack((x, point))
        y = np.append(y, task(point))
        write_trajectory(partial, task_t, x, y)
    result = write_trajectory(destination, task_t, x, y)
    partial.unlink(missing_ok=True)
    return result
