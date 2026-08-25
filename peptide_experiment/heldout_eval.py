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

import dataclasses
import subprocess

from .config import ExperimentConfig
from .optformer_optimization import run_optformer_bo
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
            elif arm.startswith("MTBO-"):
                milestone = int(arm.split("-", 1)[1])
                init_path, scores_path = build_mutation_init(cfg, task_idx, out_dir)
                run_cfg = dataclasses.replace(cfg, use_pretrained_vae=True)
                run_bo(
                    run_cfg,
                    task_idx,
                    out_dir,
                    run_id=run_id,
                    init_path=init_path,
                    scores_path=scores_path,
                    pretrained_surrogate_path=cfg.mtbo_checkpoint_dir(milestone),
                )
            elif arm.startswith("OptFormer-"):
                milestone = int(arm.split("-", 1)[1])
                run_optformer_bo(
                    cfg,
                    task_idx,
                    out_dir,
                    run_id=run_id,
                    milestone=milestone,
                    checkpoint_path=cfg.optformer_checkpoint_dir(milestone),
                )
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
            if arm == "STBO" or arm.startswith("MTBO-") or arm.startswith("OptFormer-"):
                # Neither MTBO (a different surrogate, not a different
                # initializer) nor OptFormer (its checkpoint only drives
                # subsequent proposals, conditioned on a growing trial
                # history it doesn't have yet at init time) has a checkpoint
                # that can zero-shot an init pool -- same random-mutation
                # init as STBO, an expected, correct property, not a bug
                # (see run_heldout_eval()'s branches for where each arm's
                # real method actually shows up).
                build_mutation_init(cfg, task_idx, out_dir)
            else:
                model_path = _checkpoint_for_arm(cfg, arm)
                sample_and_build_init(cfg, model_path, task_idx, out_dir)
            n_ok += 1
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"[{run_id} task {task_idx}] FAILED, continuing with rest of sweep: {e}")
            n_failed += 1

    print(f"[{run_id}] done: {n_ok} ok, {n_failed} failed, out of {len(tasks)} tasks")
