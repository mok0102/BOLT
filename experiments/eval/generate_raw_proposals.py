"""Prerequisite step for the rest of experiments/eval/: for every (arm,
milestone) entry in a manifest, sample raw candidates from that model's
checkpoint against a fixed task set, writing <task_id>_sampled_attempt*.jsonl
(+ a fixed-size init pool, unused downstream) under this package's own
raw-generation convention:
    <run_dir>/eval_raw/<task_set>/<arm>-<milestone>/

This is the only script in experiments/eval/ that touches a manifest entry's
checkpoint_dir or invokes LLM sampling -- incumbent_vs_pool_size.py and
fixed_target_rejection_bo.py are pure post-hoc analysis over whatever this
script (or a prior run of it) has already produced.

Usage (run from the BOLT repo root):
    python experiments/eval/generate_raw_proposals.py \\
        --domain peptide \\
        --config peptide_experiment/configs/peptide_poc20_bolt.yaml \\
        --manifest experiments/eval/manifests/main_bolt_vs_orpt_mi.yaml
"""

from __future__ import annotations

import argparse

from common import ModelSpec, load_manifest, raw_dir_for
from domains import DOMAINS, Domain


def generate_for_spec(domain: Domain, cfg, spec: ModelSpec, task_set_names: list[str]) -> None:
    if spec.checkpoint_dir is None:
        print(f"[generate_raw_proposals] {spec.arm}-{spec.milestone}: no checkpoint_dir in "
              "manifest, skipping (raw proposals must already exist)")
        return
    for task_set in task_set_names:
        work_dir = raw_dir_for(spec, task_set)
        task_ids = domain.task_indices(cfg, task_set)
        print(f"[generate_raw_proposals] {spec.arm}-{spec.milestone}/{task_set}: "
              f"{len(task_ids)} tasks -> {work_dir}")
        for task_id in task_ids:
            domain.sample_raw_proposals(cfg, spec.checkpoint_dir, task_id, work_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="peptide", choices=sorted(DOMAINS))
    parser.add_argument("--config", required=True, help="Any one config sharing this "
                         "manifest's similarity_threshold/init_size/task universe")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-sets", default="heldout")
    args = parser.parse_args()

    domain = DOMAINS[args.domain]
    cfg = domain.load_config(args.config)
    specs = load_manifest(args.manifest)
    task_set_names = [t.strip() for t in args.task_sets.split(",") if t.strip()]

    for spec in specs:
        generate_for_spec(domain, cfg, spec, task_set_names)


if __name__ == "__main__":
    main()
