"""Deterministic Branin task split; a task is identified by its sampled t."""

from __future__ import annotations

import numpy as np


def sample_task_values(num_train: int, num_heldout: int, seed: int) -> tuple[list[float], list[float]]:
    if num_train < 1 or num_heldout < 1:
        raise ValueError("num_train and num_heldout must be positive")
    values = np.random.default_rng(seed).uniform(0.0, 1.0, num_train + num_heldout)
    return values[:num_train].tolist(), values[num_train:].tolist()


def task_name(index: int) -> str:
    return f"task_{index:04d}"
