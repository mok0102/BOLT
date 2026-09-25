"""Step 0 of the eval chain: for every (arm, milestone) in an arm-spec file,
sample raw candidates from that model's checkpoint against a fixed task set.

Writes <task_id>_sampled_attempt*.jsonl (plus a fixed-size init pool that
nothing downstream reads) under core.arms.raw_dir_for's convention:
    <run_dir>/eval_raw/<task_set>/<arm>-<milestone>/

This is the only module in eval2 that touches a checkpoint or invokes LLM
sampling. incumbent.py and fixed_target_bo.py are post-hoc analysis over
whatever this module (or a prior run of it) already produced.
"""

from __future__ import annotations

from ..core.arms import ArmSpec, raw_dir_for


def generate_for_spec(dom, cfg, spec: ArmSpec, task_set_names: list[str]) -> None:
    if spec.checkpoint_dir is None:
        print(f"[eval2.generate_raw] {spec.arm}-{spec.milestone}: self-seeding arm with no "
              "checkpoint, skipping (it builds its own init pool at BO time)")
        return

    # One resident copy of this (arm, milestone)'s checkpoint for the whole
    # spec instead of a fresh checkpoint load per task (mi_orpt/
    # warm_sampling_pool.py; same pattern orpt.py::_sample_candidate_pool_
    # parallel already uses) -- run_eval.sh pins this process to exactly one
    # GPU via --cuda-visible-devices, so a single-entry pool covers it.
    # cfg.cuda_visible_devices being unset (ad-hoc/unsharded invocation)
    # falls back to the pre-existing per-task cold-load path.
    pools = None
    sampling_pool = None
    if cfg.cuda_visible_devices is not None:
        from peptide_experiment.mi_orpt.warm_sampling_pool import create_sampling_pools, shutdown_sampling_pools

        gpu = cfg.cuda_visible_devices
        pools = create_sampling_pools([gpu], cfg.bolt_root / "fine-tuning" / "peptides", spec.checkpoint_dir)
        sampling_pool = pools[gpu]

    try:
        for task_set in task_set_names:
            work_dir = raw_dir_for(spec, task_set)
            task_ids = dom.task_indices(cfg, task_set)
            print(f"[eval2.generate_raw] {spec.arm}-{spec.milestone}/{task_set}: "
                  f"{len(task_ids)} tasks -> {work_dir}")
            for task_id in task_ids:
                dom.sample_raw_proposals(cfg, spec.checkpoint_dir, task_id, work_dir, sampling_pool=sampling_pool)
    finally:
        if pools is not None:
            shutdown_sampling_pools(pools)


def run(dom, cfg, specs: list[ArmSpec], task_sets: list[str]) -> None:
    for spec in specs:
        generate_for_spec(dom, cfg, spec, task_sets)
