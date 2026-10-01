"""POGPE/SGPE: frozen task experts, with an online target expert for SGPE."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .branin import BraninTask
from .config import ExperimentConfig
from .steps import _scaled, _uniform_points, write_trajectory
from .task_splits import task_name


def train_gp_expert_pool(cfg: ExperimentConfig, milestone: int) -> Path:
    """Persist separate task experts; only training trajectories are read."""
    if milestone not in cfg.milestones:
        raise ValueError(f"unknown milestone {milestone}")
    root = cfg.gp_expert_dir(milestone)
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for index in range(milestone):
        source = cfg.trajectories_dir / f"{task_name(index)}.csv"
        if not source.exists():
            raise FileNotFoundError(source)
        frame = pd.read_csv(source).nlargest(cfg.gp_expert_top_n_per_task, "train_y")
        if len(frame) < 2:
            raise ValueError(f"{source}: need at least two observations")
        target = root / f"expert_{index:04d}.npz"
        refresh = not target.exists()
        if not refresh:
            with np.load(target) as saved:
                refresh = (len(saved['y']) != len(frame) or
                           abs(float(saved['lengthscale']) - cfg.bo_lengthscale) > 1e-12)
        if refresh:
            x = np.asarray([json.loads(value) for value in frame.train_x], dtype=float)
            y = frame.train_y.to_numpy(dtype=float)
            np.savez(target, x=x, y=y, lengthscale=cfg.bo_lengthscale)
        entries.append(str(target))
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"milestone": milestone, "top_n": cfg.gp_expert_top_n_per_task,
                                    "lengthscale": cfg.bo_lengthscale, "experts": entries}, indent=2))
    return manifest


class _Expert:
    def __init__(self, path: Path | None, lengthscale: float, *, x: np.ndarray | None = None,
                 y: np.ndarray | None = None):
        if path is not None:
            with np.load(path) as data:
                x = data['x']
                y = data['y']
                if abs(float(data['lengthscale']) - lengthscale) > 1e-12:
                    raise ValueError(f"{path}: lengthscale changed; retrain experts")
        if x is None or y is None or len(x) < 2:
            raise ValueError("an expert needs at least two observations")
        self.x = _scaled(x)
        self.center = float(y.mean())
        self.scale = max(float(y.std()), 1.0)
        kernel = np.exp(-0.5 * np.sum((self.x[:, None] - self.x[None, :]) ** 2, axis=-1) / lengthscale**2)
        self.chol = np.linalg.cholesky(kernel + 1e-5 * np.eye(len(self.x)))
        self.alpha = np.linalg.solve(self.chol.T, np.linalg.solve(self.chol, (y - self.center) / self.scale))
        self.lengthscale = lengthscale

    def predict(self, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sc = _scaled(candidates)
        cross = np.exp(-0.5 * np.sum((self.x[:, None] - sc[None, :]) ** 2, axis=-1) / self.lengthscale**2)
        solved = np.linalg.solve(self.chol, cross)
        mean = self.center + self.scale * (cross.T @ self.alpha)
        variance = self.scale**2 * np.maximum(1.0 - np.sum(solved**2, axis=0), 1e-6)
        return mean, variance


def run_pogpe_bo(cfg: ExperimentConfig, task_t: float, destination: Path, *, seed: int,
                 milestone: int, sgpe: bool = False) -> Path:
    if destination.exists():
        return destination
    manifest_path = cfg.gp_expert_dir(milestone) / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f"Train POGPE experts first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest['milestone'] != milestone or manifest['top_n'] != cfg.gp_expert_top_n_per_task or manifest['lengthscale'] != cfg.bo_lengthscale:
        raise ValueError(f"{manifest_path}: configuration differs from trained experts")
    experts = [_Expert(Path(path), cfg.bo_lengthscale) for path in manifest['experts']]
    if len(experts) != milestone:
        raise ValueError(f"{manifest_path}: expected {milestone} experts, got {len(experts)}")
    rng = np.random.default_rng(seed)
    x = _uniform_points(rng, cfg.init_size)
    task = BraninTask(task_t)
    y = np.asarray(task(x), dtype=float)
    target_expert = _Expert(None, cfg.bo_lengthscale, x=x, y=y) if sgpe else None
    partial = destination.with_suffix('.partial.csv')
    write_trajectory(partial, task_t, x, y)
    for _ in range(cfg.oracle_budget):
        candidates = _uniform_points(rng, cfg.bo_candidate_pool_size)
        predictions = [expert.predict(candidates) for expert in experts]
        if sgpe:
            # Match peptide SGPE: freeze the target expert fitted to the init pool.
            # Its weight equals that of the complete pretrained pool.
            predictions.append(target_expert.predict(candidates))
            weights = np.asarray([1.0 / len(experts)] * len(experts) + [1.0])
        else:
            weights = np.full(len(experts), 1.0 / len(experts))
        precision = weights[:, None] / np.asarray([var for _, var in predictions])
        combined_var = 1.0 / precision.sum(axis=0)
        combined_mean = combined_var * sum(mean * weight for (mean, _), weight in zip(predictions, precision))
        acquisition = combined_mean + cfg.bo_ucb_beta * np.sqrt(combined_var)
        # Avoid proposing a point already evaluated on this heldout task.
        distance = np.linalg.norm(_scaled(candidates)[:, None] - _scaled(x)[None, :], axis=-1)
        acquisition[np.min(distance, axis=1) < 1e-6] = -np.inf
        point = candidates[int(np.argmax(acquisition))]
        x = np.vstack([x, point])
        y = np.append(y, task(point))
        write_trajectory(partial, task_t, x, y)
    result = write_trajectory(destination, task_t, x, y)
    partial.unlink(missing_ok=True)
    return result


def run_sgpe_bo(cfg: ExperimentConfig, task_t: float, destination: Path, *, seed: int,
                milestone: int) -> Path:
    return run_pogpe_bo(cfg, task_t, destination, seed=seed, milestone=milestone, sgpe=True)
