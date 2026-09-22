"""Arm identity for eval2: which (arm, milestone) pairs a comparison covers,
where each one's run outputs live, and which checkpoint (if any) backs it.

A comparison is one yaml; a GPU shard is a pair of CLI filters (--arms /
--milestones) applied to it, rather than its own hand-copied file.

An arm is deliberately NOT resolved from a single cfg: a comparison routinely
spans models trained under different experiment_id/run_dir trees (BOLT from
peptide_main_bolt.yaml, ORPT-H1 from peptide_main_orpt_h1.yaml), which a
(cfg, arm-literal) lookup cannot express.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

BOLT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    milestone: int
    run_dir: Path
    checkpoint_dir: Path | None = None  # None for arms that self-seed (STBO/MTBO/POGPE/SGPE/LLAMBO)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else BOLT_ROOT / path


def load_arms(
    path,
    arms: list[str] | None = None,
    milestones: list[int] | None = None,
) -> list[ArmSpec]:
    """Cross-product each entry's arm name with its milestone list.

    Per-entry `milestones` overrides the file's `default_milestones` (STBO and
    LLAMBO are milestone-independent and use [0]; POGPE/SGPE overload the
    milestone field to mean n_experts). Per-entry `run_dir` overrides
    `default_run_dir`. `checkpoint_template` is rendered with {run_dir} and
    {milestone}; an entry without one is a self-seeding arm with no checkpoint.

    arms/milestones filter the result -- this is how the eval launcher shards
    work across GPUs. A filter that matches nothing raises rather than
    silently producing an empty sweep.
    """
    path = _resolve(str(path))
    raw = yaml.safe_load(path.read_text())

    default_milestones = [int(m) for m in raw["default_milestones"]]
    default_run_dir = raw.get("default_run_dir")

    specs: list[ArmSpec] = []
    seen: set[tuple[str, int]] = set()
    for entry in raw["arms"]:
        name = entry["name"]
        run_dir_str = entry.get("run_dir", default_run_dir)
        if run_dir_str is None:
            raise ValueError(f"arm {name!r} in {path} has no run_dir and the file sets no default_run_dir")
        run_dir = _resolve(run_dir_str)
        entry_milestones = [int(m) for m in entry.get("milestones", default_milestones)]
        template = entry.get("checkpoint_template")

        for milestone in entry_milestones:
            key = (name, milestone)
            if key in seen:
                raise ValueError(f"duplicate (arm, milestone) in {path}: {key}")
            seen.add(key)
            checkpoint_dir = None
            if template:
                checkpoint_dir = _resolve(template.format(run_dir=run_dir_str, milestone=milestone))
            specs.append(ArmSpec(arm=name, milestone=milestone, run_dir=run_dir, checkpoint_dir=checkpoint_dir))

    if arms is not None:
        wanted = set(arms)
        unknown = wanted - {s.arm for s in specs}
        if unknown:
            raise ValueError(f"--arms named {sorted(unknown)}, not in {path}")
        specs = [s for s in specs if s.arm in wanted]
    if milestones is not None:
        wanted_m = set(milestones)
        specs = [s for s in specs if s.milestone in wanted_m]
        if not specs:
            raise ValueError(f"--milestones {sorted(wanted_m)} matched no arm in {path}")
    return specs


def raw_dir_for(spec: ArmSpec, task_set: str) -> Path:
    """Where generate_raw writes one arm's raw LLM proposals, and where every
    later stage reads them from. task_set validity is the domain's business
    (domains.peptide.task_indices raises on an unknown one)."""
    return spec.run_dir / "eval_raw" / task_set / f"{spec.arm}-{spec.milestone}"


def bo_work_dir_for(spec: ArmSpec, task_set: str, target_pool_size: int) -> Path:
    """Where fixed_target_bo builds one arm's init pool and writes its dense
    per-task BO trajectory CSVs. figures/main_bo.py reads back from here, so
    this layout is a contract between the two -- keep them in sync."""
    return spec.run_dir / "eval_fixed_target_bo" / task_set / f"{spec.arm}-{spec.milestone}__target{target_pool_size}"
