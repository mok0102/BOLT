"""Synthetic task families used by BOLT-style experiments."""

from .branin import (
    BRANIN_BOUNDS,
    BraninParameters,
    BraninTask,
    branin,
    sample_branin_tasks,
)

__all__ = [
    "BRANIN_BOUNDS",
    "BraninParameters",
    "BraninTask",
    "branin",
    "sample_branin_tasks",
]
