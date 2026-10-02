from __future__ import annotations

from .config import ExperimentConfig
from .steps import run_bo


def checkpoint_for_arm(cfg: ExperimentConfig, arm: str):
    kind, value = arm.split("-", 1)
    return cfg.orpt_checkpoint_dir(int(value)) if kind == "ORPT" else cfg.milestone_checkpoint_dir(int(value))


def run_heldout_eval(cfg: ExperimentConfig, arm: str) -> None:
    kind, _, milestone_text = arm.partition("-")
    if kind in {"STBO", "LLAMBO"} and not milestone_text:
        milestone = None
    elif milestone_text.isdigit():
        milestone = int(milestone_text)
        if milestone not in cfg.milestones:
            raise ValueError(f"milestone {milestone} is not configured")
    else:
        raise ValueError(f"expected BOLT-50, MTBO-50, POGPE-50, SGPE-50, OptFormer-50, STBO, or LLAMBO; got {arm!r}")
    if kind == "MTBO":
        from .mtbo import train_mtbo_surrogate
        train_mtbo_surrogate(cfg, milestone)
    elif kind in {"POGPE", "SGPE"}:
        from .gp_expert_transfer import train_gp_expert_pool
        train_gp_expert_pool(cfg, milestone)
    elif kind == "OptFormer" and not cfg.optformer_checkpoint_dir(milestone).exists():
        raise FileNotFoundError(f"Train OptFormer first: {cfg.optformer_checkpoint_dir(milestone)}")
    elif kind not in {"BOLT", "ORPT", "STBO", "MTBO", "POGPE", "SGPE", "OptFormer", "LLAMBO"}:
        raise ValueError(f"unknown arm {arm!r}")
    checkpoint = checkpoint_for_arm(cfg, arm) if kind in {"BOLT", "ORPT"} else None
    if checkpoint is not None and not checkpoint.exists():
        raise RuntimeError(f"Missing checkpoint for {arm}: {checkpoint}")
    out = cfg.heldout_dir / arm
    for index, task in enumerate(cfg.heldout_tasks):
        destination = out / f"{task.name}.csv"
        seed = cfg.bo_seed + 100_000 + index
        if kind == "MTBO":
            from .mtbo import run_mtbo_bo
            run_mtbo_bo(cfg, task, destination, seed=seed, milestone=milestone)
        elif kind == "POGPE":
            from .gp_expert_transfer import run_pogpe_bo
            run_pogpe_bo(cfg, task, destination, seed=seed, milestone=milestone)
        elif kind == "SGPE":
            from .gp_expert_transfer import run_sgpe_bo
            run_sgpe_bo(cfg, task, destination, seed=seed, milestone=milestone)
        elif kind == "OptFormer":
            from .optformer import run_optformer_bo
            run_optformer_bo(cfg, task, destination, seed=seed, milestone=milestone)
        elif kind == "LLAMBO":
            from .llambo_optimization import run_llambo_bo
            run_llambo_bo(cfg, task, destination, seed=seed)
        else:
            run_bo(cfg, task, destination, seed=seed, checkpoint=checkpoint)
