"""Evaluate each milestone's own checkpoint (BOLT-<m> and ORPT-<m>) on a
fixed set of peptides it was already trained on -- the "trained peptide"
side of the constraint-violation-rate comparison (the other side is the
existing heldout20 init_only eval, unchanged).

Mirrors heldout_eval.run_init_only_eval()'s use of sample_and_build_init(),
just pointed at train-task indices instead of heldout20/heldout100, so it
reuses the exact same sampling/scoring/feasibility-patch pipeline rather
than reimplementing any of it. See
/root/.claude/plans/orpt-beta-0-25-parsed-turing.md for why this needs to be
a new eval path (nothing in peptide_experiment evaluates a milestone against
its own training tasks today) and why the fixed task set is
range(min(milestones)) rather than a resampled-per-milestone set (reuses the
same set at every milestone for an apples-to-apples curve, matching how
heldout20 is a fixed set reused across milestones).

Usage (run from the BOLT repo root):
    python experiments/constraint_violation/run_trainset_eval.py \\
        --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BOLT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOLT_ROOT))

from peptide_experiment.config import ExperimentConfig, load_config  # noqa: E402
from peptide_experiment.steps import sample_and_build_init  # noqa: E402

ARM_CHECKPOINT_FN = {
    "BOLT": ExperimentConfig.milestone_checkpoint_dir,
    "ORPT": ExperimentConfig.orpt_checkpoint_dir,
}


def train_task_subset(cfg: ExperimentConfig) -> list[int]:
    """Fixed task-index set reused at every milestone: all tasks trained on
    by the smallest milestone (see module docstring for why this is fixed
    rather than resampled per milestone)."""
    return list(range(min(cfg.milestones)))


def run_trainset_eval(cfg: ExperimentConfig, arm_prefixes: list[str]) -> None:
    train_tasks = train_task_subset(cfg)
    print(f"[trainset_eval] fixed task set: {train_tasks}")

    for milestone in cfg.milestones:
        for arm_prefix in arm_prefixes:
            checkpoint_fn = ARM_CHECKPOINT_FN[arm_prefix]
            checkpoint_dir = checkpoint_fn(cfg, milestone)
            if not checkpoint_dir.exists():
                print(f"[trainset_eval] {arm_prefix}-{milestone}: checkpoint not found at "
                      f"{checkpoint_dir}, skipping")
                continue

            out_dir = cfg.run_dir / "trainset_eval" / f"{arm_prefix}-{milestone}"
            run_id = f"trainset_eval-{arm_prefix}-{milestone}"
            n_ok, n_failed = 0, 0
            for task_idx in train_tasks:
                try:
                    sample_and_build_init(cfg, checkpoint_dir, task_idx, out_dir)
                    n_ok += 1
                except (subprocess.CalledProcessError, RuntimeError) as e:
                    print(f"[{run_id} task {task_idx}] FAILED, continuing with rest of sweep: {e}")
                    n_failed += 1
            print(f"[{run_id}] done: {n_ok} ok, {n_failed} failed, out of {len(train_tasks)} tasks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--arms",
        default="BOLT,ORPT",
        help="Comma-separated subset of {BOLT,ORPT} to evaluate (default: both)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    arm_prefixes = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arm_prefixes:
        assert a in ARM_CHECKPOINT_FN, f"unknown arm prefix {a!r}, expected one of {list(ARM_CHECKPOINT_FN)}"

    cfg.ensure_dirs()
    (cfg.run_dir / "trainset_eval").mkdir(parents=True, exist_ok=True)
    run_trainset_eval(cfg, arm_prefixes)


if __name__ == "__main__":
    main()
