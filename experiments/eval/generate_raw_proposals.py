"""Prerequisite step for the rest of experiments/eval/: for every (arm,
milestone) entry in a manifest, sample raw candidate peptides from that
model's checkpoint against a fixed task set, writing
task_<idx>_sampled_attempt*.jsonl (+ init.txt/scores.csv, unused downstream)
under this package's own raw-generation convention:
    <run_dir>/eval_raw/<task_set>/<arm>-<milestone>/

This is the only script in experiments/eval/ that touches a manifest entry's
checkpoint_dir or invokes LLM sampling -- incumbent_vs_pool_size.py,
fixed_target_rejection_bo.py and fixed_budget_rejection_bo.py are all pure
post-hoc analysis over whatever this script (or a prior run of it) has
already produced.

Usage (run from the BOLT repo root):
    python experiments/eval/generate_raw_proposals.py \\
        --config peptide_experiment/configs/peptide_poc20_bolt.yaml \\
        --manifest experiments/eval/manifests/poc20_four_arm.yaml
"""

from __future__ import annotations

import argparse

from common import ModelSpec, load_manifest, raw_dir_for, task_indices

from peptide_experiment.config import load_config
from peptide_experiment.steps import sample_and_build_init


def generate_for_spec(cfg, spec: ModelSpec, task_set_names: list[str]) -> None:
    if spec.checkpoint_dir is None:
        print(f"[generate_raw_proposals] {spec.arm}-{spec.milestone}: no checkpoint_dir in "
              "manifest, skipping (raw proposals must already exist)")
        return
    for task_set in task_set_names:
        work_dir = raw_dir_for(spec, task_set)
        indices = task_indices(cfg, task_set)
        print(f"[generate_raw_proposals] {spec.arm}-{spec.milestone}/{task_set}: "
              f"{len(indices)} tasks -> {work_dir}")
        for task_idx in indices:
            sample_and_build_init(cfg, spec.checkpoint_dir, task_idx, work_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Any one config sharing this "
                         "manifest's similarity_threshold/init_size/task universe")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-sets", default="trainset,heldout")
    args = parser.parse_args()

    cfg = load_config(args.config)
    specs = load_manifest(args.manifest)
    task_set_names = [t.strip() for t in args.task_sets.split(",") if t.strip()]

    for spec in specs:
        generate_for_spec(cfg, spec, task_set_names)


if __name__ == "__main__":
    main()
