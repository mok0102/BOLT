"""Evaluate one arm (a BOLT-<milestone> checkpoint, or STBO) independently on
every task in a held-out set. Continues past individual task failures
instead of aborting the whole sweep (known failure mode from prior smoke
testing: low-diversity LLM output at BOLT-0-like checkpoints causing hard
crashes) -- a missing task just won't have a CSV, which aggregate.py should
tolerate.

Two distinct eval procedures live here, targeting two distinct paper
artifacts (confirmed via Appendix D.6, p.24):
- run_heldout_eval(): a full BO run per task (20,000 oracle calls) -- what
  Figure 1/2's scaling curve actually reports (a "standard STBO loop
  proceeds unchanged" after initialization, per Figure 1's own caption).
- run_init_only_eval(): builds only the init pool per task, no BO
  acquisition at all -- what Table 11/12 actually report ("we report
  initialization only quality as a function of the first k oracle calls... k
  at the initialization stage"; k's max value equals the init-pool size
  exactly, not the 20,000-call budget). This is far cheaper than a full BO
  run and was previously conflated with run_heldout_eval's output, which
  reproduced a different, much more expensive quantity than Table 11 by
  mistake -- see imp_plan/01_peptide_reimplementation_plan.md.
"""

from __future__ import annotations

import subprocess

from .config import ExperimentConfig
from .steps import build_mutation_init, run_bo, sample_and_build_init


def _checkpoint_for_arm(cfg: ExperimentConfig, arm: str):
    """Resolves a checkpoint-backed arm name to its checkpoint dir. Accepts
    both 'BOLT-<m>' (the plain SFT checkpoint) and 'ORPT-<m>' (the DPO stage
    trained on top of it, Stage 4) -- STBO has no checkpoint and is handled
    separately by callers.
    """
    if arm.startswith("ORPT-"):
        return cfg.orpt_checkpoint_dir(int(arm.split("-", 1)[1]))
    assert arm.startswith("BOLT-"), f"expected an arm like 'BOLT-10' or 'ORPT-10', got {arm!r}"
    return cfg.milestone_checkpoint_dir(int(arm.split("-", 1)[1]))


def run_heldout_eval(cfg: ExperimentConfig, arm: str, task_set: str) -> None:
    out_dir = cfg.heldout_dir(task_set) / arm
    tasks = cfg.heldout_tasks(task_set)
    run_id = f"{task_set}-{arm}"

    n_ok, n_failed = 0, 0
    for task_idx in tasks:
        try:
            if arm == "STBO":
                run_bo(cfg, task_idx, out_dir, run_id=run_id, stbo=True)
            else:
                model_path = _checkpoint_for_arm(cfg, arm)
                init_path, scores_path = sample_and_build_init(
                    cfg, model_path, task_idx, out_dir
                )
                run_bo(
                    cfg,
                    task_idx,
                    out_dir,
                    run_id=run_id,
                    init_path=init_path,
                    scores_path=scores_path,
                )
            n_ok += 1
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"[{run_id} task {task_idx}] FAILED, continuing with rest of sweep: {e}")
            n_failed += 1

    print(f"[{run_id}] done: {n_ok} ok, {n_failed} failed, out of {len(tasks)} tasks")


def run_init_only_eval(cfg: ExperimentConfig, arm: str, task_set: str) -> None:
    """No-BO milestone eval (paper's Table 11/12): build each held-out task's
    init pool only (mutation-based for STBO, LLM-sampled from the milestone
    checkpoint for BOLT-<m>) -- no run_bo() call at all. aggregate.py's
    build_no_bo_milestone_eval() reads the resulting task_XXXX_scores.csv
    files directly.
    """
    out_dir = cfg.heldout_dir(task_set) / "init_only" / arm
    tasks = cfg.heldout_tasks(task_set)
    run_id = f"{task_set}-init_only-{arm}"

    n_ok, n_failed = 0, 0
    for task_idx in tasks:
        try:
            if arm == "STBO":
                build_mutation_init(cfg, task_idx, out_dir)
            else:
                model_path = _checkpoint_for_arm(cfg, arm)
                sample_and_build_init(cfg, model_path, task_idx, out_dir)
            n_ok += 1
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"[{run_id} task {task_idx}] FAILED, continuing with rest of sweep: {e}")
            n_failed += 1

    print(f"[{run_id}] done: {n_ok} ok, {n_failed} failed, out of {len(tasks)} tasks")
