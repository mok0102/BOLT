"""Evaluate one BOLT-<milestone> arm independently on every workload in the
99-task held-out set (Table 1). Continues past individual task failures
instead of aborting the whole sweep -- a multi-day sweep over a live
Postgres instance will hit transient failures, mirrors
peptide_experiment/heldout_eval.py's discipline.

Only one eval tier exists for this domain (unlike peptide's heldout20/
heldout100 + init-only-vs-full-BO split) -- nothing in the paper facts this
stage targets suggests a query-plan analog of peptide's Table-11
init-only-quality ablation. See imp_plan/02_query_plan_reimplementation_plan.md.
"""

from __future__ import annotations

import subprocess

from .config import ExperimentConfig
from .steps import run_bo, sample_and_build_init


def _checkpoint_for_arm(cfg: ExperimentConfig, arm: str):
    if arm.startswith("ORPT-") and cfg.build_orpt:
        return cfg.orpt_checkpoint_dir(int(arm.split("-", 1)[1]))
    assert arm.startswith("BOLT-"), f"expected BOLT-<m> or enabled ORPT-<m>, got {arm!r}"
    return cfg.milestone_checkpoint_dir(int(arm.split("-", 1)[1]))


def run_heldout_eval(cfg: ExperimentConfig, arm: str) -> None:
    out_dir = cfg.heldout_dir / arm
    workloads = cfg.heldout_tasks
    run_id = f"heldout-{arm}"
    first_milestone = cfg.milestones[0]

    n_ok, n_failed = 0, 0
    for workload in workloads:
        try:
            if arm == f"BOLT-{first_milestone}":
                # Same convention as trajectory_chain.py's pre-first-
                # milestone case: use the LOLBO script's own init_w_bao=True
                # default, no LLM-checkpoint sampling involved.
                run_bo(cfg, workload, out_dir, run_id=run_id)
            else:
                model_path = _checkpoint_for_arm(cfg, arm)
                init_csv_path = sample_and_build_init(cfg, model_path, workload, out_dir)
                run_bo(cfg, workload, out_dir, run_id=run_id, init_csv_path=init_csv_path, init_w_llm=True)
            n_ok += 1
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"[{run_id} {workload}] FAILED, continuing with rest of sweep: {e}")
            n_failed += 1

    print(f"[{run_id}] done: {n_ok} ok, {n_failed} failed, out of {len(workloads)} workloads")
