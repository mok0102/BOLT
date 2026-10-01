"""Branin synthetic objective with task variation only in ``t``.

The task family is

    f_t(x1, x2) = a (x2 - b*x1**2 + c*x1 - r)**2
                  + s*(1 - t)*cos(x1) + s,

where ``a``, ``b``, ``c``, ``r`` and ``s`` are fixed to the canonical
Branin values and each task draws ``t ~ Uniform(0, 1)``.  The raw Branin
function is a minimization objective.  ``BraninTask`` returns its negative
by default so it follows the peptide experiment's maximize-score convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


BRANIN_BOUNDS: tuple[tuple[float, float], tuple[float, float]] = (
    (-5.0, 10.0),
    (0.0, 15.0),
)


@dataclass(frozen=True)
class BraninParameters:
    """Fixed parameters shared by every task in the family."""

    a: float = 1.0
    b: float = 5.1 / (4.0 * np.pi**2)
    c: float = 5.0 / np.pi
    r: float = 6.0
    s: float = 10.0


DEFAULT_PARAMETERS = BraninParameters()


def _validate_task_t(task_t: float) -> float:
    task_t = float(task_t)
    if not np.isfinite(task_t) or not 0.0 <= task_t <= 1.0:
        raise ValueError(f"task_t must be finite and in [0, 1], got {task_t!r}")
    return task_t


def branin(
    x: ArrayLike,
    task_t: float,
    *,
    parameters: BraninParameters = DEFAULT_PARAMETERS,
) -> float | NDArray[np.float64]:
    """Evaluate the raw (minimized) Branin objective for one task.

    ``x`` may be one point with shape ``(2,)`` or a batch with shape
    ``(..., 2)``.  A single point returns ``float``; a batch returns an array
    with the leading shape preserved.  Bounds are exposed as
    :data:`BRANIN_BOUNDS` but are not enforced, which keeps the function
    useful to optimizers that temporarily evaluate outside the box.
    """

    task_t = _validate_task_t(task_t)
    points = np.asarray(x, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 2:
        raise ValueError(f"x must have shape (2,) or (..., 2), got {points.shape}")

    x1 = points[..., 0]
    x2 = points[..., 1]
    p = parameters
    values = (
        p.a * (x2 - p.b * x1**2 + p.c * x1 - p.r) ** 2
        + p.s * (1.0 - task_t) * np.cos(x1)
        + p.s
    )
    if values.ndim == 0:
        return float(values)
    return values


@dataclass(frozen=True)
class BraninTask:
    """Callable task object analogous to a peptide task-specific oracle."""

    task_t: float
    maximize: bool = True
    parameters: BraninParameters = DEFAULT_PARAMETERS

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_t", _validate_task_t(self.task_t))

    @property
    def task_id(self) -> str:
        """Stable, human-readable task context for prompts and result files."""

        return f"task_t={self.task_t:.17g}"

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return BRANIN_BOUNDS

    def raw(self, x: ArrayLike) -> float | NDArray[np.float64]:
        """Return the conventional minimization value ``f_t(x)``."""

        return branin(x, self.task_t, parameters=self.parameters)

    def __call__(self, x: ArrayLike) -> float | NDArray[np.float64]:
        """Return a score; by default larger is better, as in peptide BOLT."""

        values = self.raw(x)
        return -values if self.maximize else values


def sample_branin_tasks(
    num_tasks: int,
    *,
    seed: int | None = None,
    maximize: bool = True,
) -> list[BraninTask]:
    """Draw reproducible tasks with independent ``t ~ Uniform[0, 1)``."""

    if isinstance(num_tasks, bool) or not isinstance(num_tasks, int) or num_tasks < 0:
        raise ValueError(f"num_tasks must be a non-negative integer, got {num_tasks!r}")
    rng = np.random.default_rng(seed)
    return [BraninTask(task_t=t, maximize=maximize) for t in rng.uniform(0.0, 1.0, num_tasks)]


def task_t_values(tasks: Sequence[BraninTask]) -> NDArray[np.float64]:
    """Return task contexts as an array, useful when saving a task split."""

    return np.asarray([task.task_t for task in tasks], dtype=np.float64)
