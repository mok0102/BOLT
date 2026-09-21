"""Per-domain constants this package needs, hardcoded once rather than
loaded via peptide_experiment.config/llvm_experiment.config -- see the plan
file's "fully independent" decision. These match the values decided this
session (2026-09-21): both domains share target_pool_size=100; oracle
budgets and the fig:scaling representative-k values come straight from
paper/experiments.tex's own RESULT FIXTURE comments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainSpec:
    oracle_budget: int
    target_pool_size: int
    milestones: tuple[int, ...]
    scaling_k: int  # fig:scaling panel (a)'s representative k, per experiments.tex
    main_bo_arms: tuple[str, ...]  # arms fig:main-bo/fewshot/scaling expect (paper-label space)


DOMAIN_SPECS: dict[str, DomainSpec] = {
    "peptide": DomainSpec(
        oracle_budget=500,
        target_pool_size=100,
        milestones=(10, 20, 50, 250, 400, 500, 600),
        scaling_k=200,
        main_bo_arms=("BOLT", "ORPT", "STBO", "MTBO", "POGPE", "SGPE", "OptFormer", "LLAMBO"),
    ),
    "llvm": DomainSpec(
        oracle_budget=200,
        target_pool_size=100,
        milestones=(10, 20, 50, 250, 400, 500, 600),
        scaling_k=50,
        main_bo_arms=("BOLT", "ORPT", "STBO", "MTBO", "POGPE", "SGPE", "OptFormer", "LLAMBO"),
    ),
}


def spec_for(domain: str) -> DomainSpec:
    try:
        return DOMAIN_SPECS[domain]
    except KeyError as e:
        raise KeyError(f"unknown domain {domain!r}, expected one of {list(DOMAIN_SPECS)}") from e
