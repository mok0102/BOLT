"""Thin re-export of synthetic_experiment.global_optimum (where the actual
verification search now lives, since the task-manifest BUILDER needs it as
a core-library dependency -- see synthetic_experiment/task_splits.py),
plus an independent re-audit against a frozen manifest: re-run the search
with different seeds than the manifest was built with and confirm
agreement, a "trust but verify" check consistent with this project's own
rule of never silently assuming an analytic optimum.
"""

from __future__ import annotations

from synthetic_experiment.global_optimum import (
    VerifiedOptimum,
    edge_margin,
    is_well_posed,
    verify_optimum,
)
from synthetic_experiment.task_splits import TaskManifest, TaskRecord

__all__ = ["VerifiedOptimum", "edge_margin", "is_well_posed", "verify_optimum", "audit_manifest"]


def audit_manifest(
    manifest: TaskManifest,
    *,
    split: str,
    seed_offset: int = 1_000_000,
    f_star_atol: float = 1e-6,
    x_star_atol: float = 1e-3,
) -> list[tuple[TaskRecord, VerifiedOptimum, bool]]:
    """Re-verify every task in `split` with a seed the manifest was NOT
    built with, and confirm the result agrees with what's already recorded.
    `x_star` is deterministic given a task's transform (derived analytically
    from the transformed canonical minimizers, not from wherever the
    stochastic search happened to converge -- see global_optimum.py), so a
    disagreement here is a real red flag, not seed noise. Raises on the
    first disagreement rather than collecting and silently reporting a
    partial pass."""
    results = []
    for record in manifest.tasks(split):
        fresh = verify_optimum(record.oracle(maximize=False), seed=seed_offset + record.index)
        agrees = (
            abs(fresh.f_star - record.verified.f_star) <= f_star_atol
            and abs(fresh.x_star[0] - record.verified.x_star[0]) <= x_star_atol
            and abs(fresh.x_star[1] - record.verified.x_star[1]) <= x_star_atol
        )
        if not agrees:
            raise ValueError(
                f"{split} task {record.index}: re-audit disagrees with the frozen manifest "
                f"(manifest f*={record.verified.f_star}, x*={record.verified.x_star}; "
                f"re-audit f*={fresh.f_star}, x*={fresh.x_star})"
            )
        results.append((record, fresh, agrees))
    return results
