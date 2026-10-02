"""Branin synthetic objective family: one canonical landscape, diversified by
a random per-task affine transform of input space (shift + rotation +
isotropic scale), not by varying the Branin formula's own constants.

Canonical Branin (``branin()``) is the textbook function, with its usual
three tied global minimizers along the valley
``x2 = b*x1**2 - c*x1 + r`` at ``x1 in {..., -3*pi, -pi, pi, 3*pi, ...}``,
achieving the standard minimum value ``f* = s*t = 0.397887...`` (using the
canonical constant ``t = 1/(8*pi)``).

A task is a :class:`BraninTaskTransform`: it moves canonical Branin's input
space around before evaluation (``BraninTask.raw`` inverts the transform,
then evaluates canonical Branin), so every task shares the exact same
landscape shape/difficulty/achievable value, differing only in where and
how that landscape sits inside the fixed :data:`BRANIN_BOUNDS` box. This
replaces an earlier single-scalar-``t`` family whose global optimum
location was independent of ``t`` (only the achieved value changed),
which let context-free "reuse points from other tasks" strategies score
deceptively well -- see ``experiments/eval2_branin_motivation``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

BRANIN_BOUNDS: tuple[tuple[float, float], tuple[float, float]] = (
    (-5.0, 10.0),
    (0.0, 15.0),
)

# Every task's transform is applied about the box's own center, so a pure
# rotation or scale (zero shift) still keeps a task's landscape centered in
# the same box every other task uses.
TRANSFORM_PIVOT: tuple[float, float] = (
    (BRANIN_BOUNDS[0][0] + BRANIN_BOUNDS[0][1]) / 2.0,
    (BRANIN_BOUNDS[1][0] + BRANIN_BOUNDS[1][1]) / 2.0,
)


@dataclass(frozen=True)
class BraninParameters:
    """Fixed parameters shared by every task in the family -- canonical
    Branin-Hoo constants, never varied per task (diversity comes entirely
    from :class:`BraninTaskTransform` instead). ``t`` fixed at the textbook
    constant ``1/(8*pi)`` so ``f_star = s*t`` matches the standard published
    Branin-Hoo global minimum, ``0.397887...`` -- not an arbitrary choice."""

    a: float = 1.0
    b: float = 5.1 / (4.0 * np.pi**2)
    c: float = 5.0 / np.pi
    r: float = 6.0
    s: float = 10.0
    t: float = 1.0 / (8.0 * np.pi)


DEFAULT_PARAMETERS = BraninParameters()
CANONICAL_F_STAR = DEFAULT_PARAMETERS.s * DEFAULT_PARAMETERS.t


def branin(
    x: ArrayLike,
    *,
    parameters: BraninParameters = DEFAULT_PARAMETERS,
) -> float | NDArray[np.float64]:
    """Evaluate the raw (minimized) CANONICAL Branin objective -- no task
    variation here; see :class:`BraninTask` for the per-task transform.

    ``x`` may be one point with shape ``(2,)`` or a batch with shape
    ``(..., 2)``. A single point returns ``float``; a batch returns an array
    with the leading shape preserved. Bounds are exposed as
    :data:`BRANIN_BOUNDS` but are not enforced, which keeps the function
    useful to optimizers that temporarily evaluate outside the box.
    """
    points = np.asarray(x, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 2:
        raise ValueError(f"x must have shape (2,) or (..., 2), got {points.shape}")

    x1 = points[..., 0]
    x2 = points[..., 1]
    p = parameters
    values = (
        p.a * (x2 - p.b * x1**2 + p.c * x1 - p.r) ** 2
        + p.s * (1.0 - p.t) * np.cos(x1)
        + p.s
    )
    if values.ndim == 0:
        return float(values)
    return values


def canonical_global_minimizers(
    parameters: BraninParameters = DEFAULT_PARAMETERS,
    n_range: range = range(-7, 10, 2),
) -> list[tuple[float, float]]:
    """The (unbounded) canonical Branin has infinitely many tied global
    minimizers, one per odd multiple of pi along x1: ``x1 = n*pi``,
    ``x2 = b*x1**2 - c*x1 + r`` (the vertex, for that x1, of the valley
    parabola -- the only x2 where the quadratic term vanishes). ``n_range``
    covers every odd n whose image could plausibly land inside
    :data:`BRANIN_BOUNDS` under this family's sampled shift/rotation/scale
    ranges; used only as an analytic pre-filter / cross-check, never
    trusted alone for acceptance (see ``task_splits.py``'s rejection loop,
    which always numerically re-verifies)."""
    p = parameters
    return [
        (float(n) * np.pi, p.b * (n * np.pi) ** 2 - p.c * (n * np.pi) + p.r)
        for n in n_range
        if n % 2 != 0
    ]


@dataclass(frozen=True)
class BraninTaskTransform:
    """A task's input-space affine transform, applied about
    :data:`TRANSFORM_PIVOT`: evaluating a task-space point ``x`` first maps
    it back to canonical space via :meth:`to_canonical`, then evaluates
    canonical :func:`branin` there. Only finiteness/sign are validated here;
    the actual sampling RANGES (how much shift/rotation/scale a *generated*
    task may have) are the task generator's concern, not this dataclass's --
    keeps e.g. :data:`IDENTITY_TRANSFORM` trivially constructible for tests
    regardless of whatever ranges the generator uses."""

    shift: tuple[float, float]
    rotation_deg: float  # counterclockwise about TRANSFORM_PIVOT, in [0, 360)
    scale: float  # > 0, isotropic, about the same pivot

    def __post_init__(self) -> None:
        shift = (float(self.shift[0]), float(self.shift[1]))
        rotation_deg = float(self.rotation_deg) % 360.0
        scale = float(self.scale)
        if not (np.isfinite(shift).all() and np.isfinite(rotation_deg) and np.isfinite(scale)):
            raise ValueError(f"BraninTaskTransform fields must be finite, got {self}")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale!r}")
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "rotation_deg", rotation_deg)
        object.__setattr__(self, "scale", scale)

    @property
    def rotation_rad(self) -> float:
        return np.deg2rad(self.rotation_deg)

    def to_canonical(self, x: ArrayLike) -> NDArray[np.float64]:
        """Task-space -> canonical-space (the evaluation direction):
        ``u = pivot + R(theta)^T (x - pivot - shift) / scale``."""
        x = np.asarray(x, dtype=np.float64)
        pivot = np.asarray(TRANSFORM_PIVOT)
        shift = np.asarray(self.shift)
        c, s = np.cos(self.rotation_rad), np.sin(self.rotation_rad)
        d = x - pivot - shift
        d1, d2 = d[..., 0], d[..., 1]
        u1 = (c * d1 + s * d2) / self.scale
        u2 = (-s * d1 + c * d2) / self.scale
        return pivot + np.stack([u1, u2], axis=-1)

    def from_canonical(self, u: ArrayLike) -> NDArray[np.float64]:
        """Canonical-space -> task-space (the inverse of :meth:`to_canonical`,
        used to map the canonical minimizers through a candidate transform
        during task generation -- never needed at evaluation time):
        ``x = pivot + scale * R(theta) (u - pivot) + shift``."""
        u = np.asarray(u, dtype=np.float64)
        pivot = np.asarray(TRANSFORM_PIVOT)
        shift = np.asarray(self.shift)
        c, s = np.cos(self.rotation_rad), np.sin(self.rotation_rad)
        e = u - pivot
        e1, e2 = e[..., 0], e[..., 1]
        x1 = self.scale * (c * e1 - s * e2)
        x2 = self.scale * (s * e1 + c * e2)
        return pivot + shift + np.stack([x1, x2], axis=-1)


IDENTITY_TRANSFORM = BraninTaskTransform(shift=(0.0, 0.0), rotation_deg=0.0, scale=1.0)


@dataclass(frozen=True)
class BraninTask:
    """Callable task object analogous to a peptide task-specific oracle."""

    transform: BraninTaskTransform
    maximize: bool = True
    parameters: BraninParameters = DEFAULT_PARAMETERS

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return BRANIN_BOUNDS

    def raw(self, x: ArrayLike) -> float | NDArray[np.float64]:
        """Return the conventional minimization value for this task."""
        return branin(self.transform.to_canonical(x), parameters=self.parameters)

    def __call__(self, x: ArrayLike) -> float | NDArray[np.float64]:
        """Return a score; by default larger is better, as in peptide BOLT."""
        values = self.raw(x)
        return -values if self.maximize else values
