"""Synthetic task families used by BOLT-style experiments."""

from .branin import (
    BRANIN_BOUNDS,
    CANONICAL_F_STAR,
    IDENTITY_TRANSFORM,
    TRANSFORM_PIVOT,
    BraninParameters,
    BraninTask,
    BraninTaskTransform,
    branin,
    canonical_global_minimizers,
)

__all__ = [
    "BRANIN_BOUNDS",
    "CANONICAL_F_STAR",
    "IDENTITY_TRANSFORM",
    "TRANSFORM_PIVOT",
    "BraninParameters",
    "BraninTask",
    "BraninTaskTransform",
    "branin",
    "canonical_global_minimizers",
]
