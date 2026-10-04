"""ORPT (Stage 4) for the mol domain: a DPO preference-tuning stage trained
on top of each milestone's own BOLT-<m> SFT checkpoint. Mirrors
peptide_experiment/orpt.py's shape closely (read-only reference, never
imported -- isolation contract): same cumulative-trajectory-data input,
same idempotent-skip-if-exists pattern, same train_orpt_milestone/
_fit_dpo_batch_size logic -- but builds preference pairs via mol's own
mi_orpt/build_pairs.py and mol_experiment.config.MolExperimentConfig
throughout.

cfg.mi_candidate_temperature (a dedicated candidate pool sampled from the
current milestone's own checkpoint via LLM sampling, at a temperature
typically higher than regular deployment-time sampling) is implemented via
mol_sample_and_build_init -- see _sample_mol_candidate_pool below.
cfg.mi_parallel_gpus, when set, dispatches this across
mi_orpt/warm_sampling_pool.py (one worker process per GPU, model loaded
once per worker and reused across every task) instead of serially on this
process's own device -- mirrors this file's own one-step-BO pool
(mi_orpt/warm_pool.py) and this repo's own reference-scoring pool
(mi_orpt/warm_scoring_pool.py). Still NOT ported: trajectory_chain.py's own
regular per-task sampling (the non-mi_orpt call to mol_sample_and_build_init)
stays serial -- see that module's docstring for why (no parallel per-task
dispatch wired into the main trajectory loop yet).

Not ported from peptide_experiment/orpt.py in this skeleton (explicitly
deferred, not silently approximated):
- orpt_pair_source == "objective_ranked": peptide's uniform-random-pairing
  baseline arm (make_dpo_train_data_csv.py) has no mol port yet -- see
  build_orpt_pairs' NotImplementedError below and
  mol_experiment/config.py's orpt_pair_source docstring for why mol
  defaults to "matched_intervention" instead.
- orpt_loss_type variants other than "dpo" (dpo_ii_penalty/bpo and their
  fa_orpt_*/orpt_bpo_alpha hyperparameters): not part of
  MolExperimentConfig (see that file's module docstring) -- asserted here
  rather than silently ignored.
"""

from __future__ import annotations

import dataclasses
import random
import sys

import yaml

from .config import MolExperimentConfig
from .mi_orpt.candidate_bank import min_bank_size_needed
from .mi_orpt.warm_sampling_pool import create_sampling_pool, sample_tasks_parallel
from .steps import _run, cleanup_intermediate_epochs, distributed_finetune_launch, materialize_hf_checkpoint, mol_sample_and_build_init

FINE_TUNING_DIR = "fine-tuning/mol"


def _sample_mol_candidate_pool(cfg: MolExperimentConfig, model_path, task_indices: list[int], pool_dir, temperature: float):
    """Runs mol_sample_and_build_init(cfg, model_path, i, pool_dir, temperature=temperature)
    for every i in task_indices. Serial on this process's own device when
    cfg.mi_parallel_gpus is unset; dispatched across
    mi_orpt/warm_sampling_pool.py otherwise. Returns
    {task_idx: (init_path, scores_path)}."""
    if not cfg.mi_parallel_gpus:
        return {i: mol_sample_and_build_init(cfg, model_path, i, pool_dir, temperature=temperature) for i in task_indices}

    sampling_pool = create_sampling_pool(cfg.mi_parallel_gpus, model_path)
    try:
        return sample_tasks_parallel(sampling_pool, cfg, model_path, task_indices, pool_dir, temperature)
    finally:
        sampling_pool.shutdown(wait=True)


def build_orpt_pairs(cfg: MolExperimentConfig, milestone: int):
    """Build preference pairs from the same cumulative trajectory data
    [0, milestone) that BOLT-<milestone>'s own SFT dataset uses.

    Only cfg.orpt_pair_source == "matched_intervention" is implemented (see
    module docstring for what's deferred and why).
    """
    pairs_csv = cfg.orpt_pairs_dir / f"orpt_pairs_{milestone}.csv"
    pairs_jsonl = cfg.orpt_pairs_dir / f"orpt_pairs_{milestone}.jsonl"

    if pairs_jsonl.exists():
        print(f"[orpt pairs {milestone}] already exists at {pairs_jsonl}, skipping")
        return pairs_jsonl

    if cfg.orpt_pair_source != "matched_intervention":
        raise NotImplementedError(
            f"cfg.orpt_pair_source={cfg.orpt_pair_source!r} is not implemented for the mol domain "
            "-- only 'matched_intervention' is (see this module's docstring and "
            "mol_experiment/config.py's orpt_pair_source docstring)."
        )
    task_indices = list(range(milestone))
    if cfg.mi_max_tasks_per_milestone is not None and len(task_indices) > cfg.mi_max_tasks_per_milestone:
        # Fixed-seed uniform subsample, not the first N -- pair construction cost
        # is linear in len(task_indices). BOLT-<milestone>'s own SFT dataset
        # (train_milestone(), a sibling call) still uses every task in
        # [0, milestone) -- only the DPO stage's pair pool is subsampled.
        task_indices = sorted(random.Random(1000 + milestone).sample(task_indices, cfg.mi_max_tasks_per_milestone))
        print(
            f"[orpt pairs {milestone}] mi_max_tasks_per_milestone={cfg.mi_max_tasks_per_milestone}: "
            f"using {len(task_indices)}/{milestone} tasks for pair construction"
        )
    input_csvs = [cfg.trajectories_csv_dir / f"task_{i:04d}.csv" for i in task_indices]

    torchtune_config_path = cfg.bolt_root / FINE_TUNING_DIR / "torchtune_config" / cfg.torchtune_config

    candidate_inits: list = []
    candidate_scores: list = []
    if cfg.mi_candidate_temperature is not None:
        # Dedicated candidate pool for mi_orpt's own bank, sampled at
        # cfg.mi_candidate_temperature from THIS milestone's own BOLT-<m> --
        # separate from trajectories_csv_dir, so BOLT-<milestone>'s own SFT
        # dataset (built from trajectories_csv_dir elsewhere) is unaffected.
        # Milestone-scoped path: a different checkpoint samples this pool at
        # every milestone, so an unscoped path would make a later milestone
        # silently reuse an earlier milestone's stale pool instead of
        # resampling (mol_sample_and_build_init only regenerates when its
        # init/scores files don't already exist).
        pool_dir = cfg.run_dir / "mi_candidate_pool" / f"milestone_{milestone:04d}"
        # Target unique-candidate count, NOT cfg.init_size: mol_sample_and_build_init's
        # own retry loop only chases cfg.init_size feasible draws -- the bank
        # needs min_bank_size_needed(...) feasible candidates AFTER dedup, so
        # the pool must be meaningfully larger than that floor. 15x mirrors
        # peptide's own validated multiplier (peptide_experiment/orpt.py's
        # own comment: accounts for a feasibility floor around ~len(pool)//10
        # plus duplicate collapse on dedup) -- not independently re-derived
        # for mol, since _ensure_mol_constraint_feasible's floor
        # (max(1, len(seqs)//10)) is the same shape as peptide's.
        pool_size = cfg.mi_candidate_pool_size or (
            min_bank_size_needed(cfg.mi_background_size or cfg.init_size, cfg.mi_max_candidates_per_task) * 15
        )
        pool_cfg = dataclasses.replace(cfg, init_size=pool_size)
        if cfg.mi_parallel_gpus:
            print(
                f"[orpt pairs {milestone}] sampling {len(task_indices)} task(s)' candidate pools across "
                f"{len(cfg.mi_parallel_gpus)} GPU(s) ({cfg.mi_parallel_gpus})"
            )
        results = _sample_mol_candidate_pool(
            pool_cfg, cfg.milestone_checkpoint_dir(milestone), task_indices, pool_dir, cfg.mi_candidate_temperature
        )
        for i in task_indices:
            init_path, scores_path = results[i]
            candidate_inits.append(init_path)
            candidate_scores.append(scores_path)

    cmd = [
        sys.executable,
        "-m",
        "mol_experiment.mi_orpt.build_pairs",
        "--input-csv",
        *input_csvs,
        "--task-index",
        *task_indices,
        "--output-csv",
        pairs_csv,
        "--output-jsonl",
        pairs_jsonl,
        "--torchtune-config",
        torchtune_config_path,
        "--checkpoint-dir",
        cfg.milestone_checkpoint_dir(milestone),
        "--tau-mol",
        cfg.tau_mol,
        "--m",
        cfg.mi_background_size or cfg.init_size,
        "--target-pairs-per-task",
        cfg.mi_target_pairs_per_task,
        "--max-candidates-per-task",
        cfg.mi_max_candidates_per_task,
        "--num-backgrounds",
        cfg.mi_num_backgrounds,
        "--tau-q",
        cfg.mi_tau_q,
        "--z-min",
        cfg.mi_z_min,
        "--delta-t",
        cfg.mi_delta_t,
        "--bo-steps",
        cfg.mi_bo_steps,
        "--milestone",
        milestone,
        "--experiment-id",
        cfg.experiment_id,
        "--bsz",
        cfg.bsz,
        "--seed",
        42,
    ]
    cmd.append("--require-h0" if cfg.mi_require_h0 else "--no-require-h0")
    if candidate_inits:
        cmd += ["--candidate-init", *candidate_inits, "--candidate-scores", *candidate_scores]
    if cfg.cuda_visible_devices is not None:
        cmd += ["--cuda-visible-devices", cfg.cuda_visible_devices]
    launch_cfg = cfg
    if cfg.mi_parallel_gpus:
        # --parallel-gpus only parallelizes reference-model scoring in mol's
        # build_pairs.py (see that file's docstring) -- still worth passing
        # through since scoring is a real, measured cost at scale.
        cmd += ["--parallel-gpus", *cfg.mi_parallel_gpus]
        launch_cfg = dataclasses.replace(cfg, cuda_visible_devices=",".join(cfg.mi_parallel_gpus))
    _run(cmd, cwd=cfg.bolt_root, cfg=launch_cfg)
    return pairs_jsonl


MIN_DPO_STEPS = 8


def _filter_pairs_by_length(cfg: MolExperimentConfig, pairs_jsonl, milestone: int):
    """Returns the pairs file the DPO stage should train on. With
    cfg.orpt_max_pair_tokens=None this is pairs_jsonl itself. Otherwise pairs
    whose longer side exceeds the limit (HF chat-template token count, the same
    measure the limit was chosen with) are dropped into
    orpt_pairs_<m>_maxtok<N>.jsonl; the original file is left untouched.
    Always reports what was dropped (count, distinct targets) -- never silent."""
    max_tokens = cfg.orpt_max_pair_tokens
    if max_tokens is None:
        return pairs_jsonl

    import json

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_checkpoint_dir)

    def n_tokens(messages) -> int:
        out = tokenizer.apply_chat_template(messages, tokenize=True)
        return len(out["input_ids"] if hasattr(out, "keys") else out)

    kept, dropped_targets = [], set()
    n_total = 0
    for line in pairs_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        n_total += 1
        rec = json.loads(line)
        if max(n_tokens(rec["chosen"]), n_tokens(rec["rejected"])) <= max_tokens:
            kept.append(line)
        else:
            user = next(m["content"] for m in rec["chosen"] if m["role"] == "user")
            dropped_targets.add(user.split("\n")[1])  # the target protein sequence line

    out_path = pairs_jsonl.with_name(f"{pairs_jsonl.stem}_maxtok{max_tokens}.jsonl")
    tmp_path = out_path.with_suffix(".tmp")
    tmp_path.write_text("".join(line + "\n" for line in kept))
    tmp_path.replace(out_path)
    print(
        f"[orpt milestone {milestone}] orpt_max_pair_tokens={max_tokens}: kept {len(kept)}/{n_total} pairs, "
        f"dropped {n_total - len(kept)} pair(s) from {len(dropped_targets)} distinct target(s) -> {out_path}"
    )
    return out_path


def _fit_dpo_batch_size(extra_overrides, pairs_jsonl, nproc_per_node, cfg, fine_tuning_dir, milestone):
    """Byte-identical port of peptide_experiment/orpt.py::_fit_dpo_batch_size
    -- see that docstring for the full torchtune-drops-incomplete-batches
    rationale (domain-generic; no mol-specific content)."""
    n_pairs = sum(1 for line in pairs_jsonl.read_text().splitlines() if line.strip())
    yaml_batch = None
    for override in extra_overrides:
        if override.startswith("batch_size="):
            yaml_batch = int(override.split("=", 1)[1])
    if yaml_batch is None:
        with open(fine_tuning_dir / "torchtune_config" / cfg.orpt_torchtune_config) as f:
            yaml_batch = yaml.safe_load(f)["batch_size"]

    per_device = yaml_batch
    while per_device > 1 and n_pairs // (per_device * nproc_per_node) < MIN_DPO_STEPS:
        per_device //= 2
    steps = n_pairs // (per_device * nproc_per_node)
    if steps == 0:
        raise RuntimeError(
            f"[orpt milestone {milestone}] {n_pairs} preference pair(s) cannot fill a single "
            f"batch across {nproc_per_node} GPU(s) even at batch_size=1 -- torchtune would drop "
            "every batch and save an untrained (BOLT-identical) ORPT checkpoint. Raise "
            "mi_target_pairs_per_task/mi_max_tasks_per_milestone, or lower mi_parallel_gpus."
        )
    print(
        f"[orpt milestone {milestone}] {n_pairs} pairs, {nproc_per_node} GPU(s): "
        f"batch_size {yaml_batch} -> {per_device}/device (global {per_device * nproc_per_node}), "
        f"{steps} optimizer step(s)/epoch x {cfg.orpt_epochs} epoch(s)"
    )
    return [o for o in extra_overrides if not o.startswith("batch_size=")] + [f"batch_size={per_device}"]


def train_orpt_milestone(cfg: MolExperimentConfig, milestone: int):
    """Train ORPT-<milestone>: a DPO stage on top of that same milestone's
    own BOLT-<milestone> checkpoint (pi_theta init == pi_ref, via
    lora_dpo_distributed's LoRA adapter-disable trick -- no separate
    ref_checkpointer needed)."""
    assert cfg.orpt_loss_type == "dpo", (
        f"cfg.orpt_loss_type={cfg.orpt_loss_type!r} -- only 'dpo' is implemented for the mol domain "
        "(dpo_ii_penalty/bpo variants are not part of MolExperimentConfig, see its module docstring)"
    )

    ckpt_dir = cfg.checkpoints_dir / f"ORPT-{milestone}"
    final_ckpt = cfg.orpt_checkpoint_dir(milestone)
    if (final_ckpt / "model.safetensors.index.json").exists():
        print(f"[orpt milestone {milestone}] checkpoint already exists at {final_ckpt}, skipping DPO")
        return final_ckpt

    bolt_ckpt = cfg.milestone_checkpoint_dir(milestone)
    if not (bolt_ckpt / "model.safetensors.index.json").exists():
        raise RuntimeError(
            f"ORPT-{milestone} requires BOLT-{milestone} to already exist at {bolt_ckpt}, "
            "but it wasn't found -- train_milestone() must run before train_orpt_milestone()"
        )

    fine_tuning_dir = cfg.bolt_root / FINE_TUNING_DIR
    pairs_jsonl = _filter_pairs_by_length(cfg, build_orpt_pairs(cfg, milestone), milestone)
    if pairs_jsonl.stat().st_size == 0:
        # matched_intervention's reliability filter can legitimately yield zero
        # pairs for every task in a milestone -- surface that clearly instead of
        # letting torchtune's DPO recipe fail on an empty dataset with an opaque
        # ChildFailedError, and instead of silently continuing (downstream
        # checkpoint_to_sample_from() has no fallback for a missing ORPT-<m>
        # when cfg.build_orpt=True).
        raise RuntimeError(
            f"[orpt milestone {milestone}] {pairs_jsonl} has 0 preference pairs -- nothing to train "
            f"ORPT-{milestone} on. Likely cause: eligible bank too small at this milestone/scale for "
            "min_bank_size_needed(m)=m+1 across every task."
        )

    overrides = [
        f"output_dir={ckpt_dir}",
        f"dataset.data_files={pairs_jsonl}",
        f"tokenizer.path={cfg.base_checkpoint_dir}/vocab.json",
        f"tokenizer.merges_file={cfg.base_checkpoint_dir}/merges.txt",
        f"checkpointer.checkpoint_dir={bolt_ckpt}",
        f"epochs={cfg.orpt_epochs}",
        f"loss.beta={cfg.orpt_beta}",
        f"optimizer.lr={cfg.orpt_lr}",
        "seed=42",
        f"metric_logger.log_dir={cfg.tensorboard_dir / f'ORPT-{milestone}'}",
    ]

    torchrun_flags, extra_overrides, launch_cfg = distributed_finetune_launch(cfg, cfg.orpt_torchtune_config, fine_tuning_dir)
    nproc_per_node = len(cfg.mi_parallel_gpus) if cfg.mi_parallel_gpus else 1
    extra_overrides = _fit_dpo_batch_size(extra_overrides, pairs_jsonl, nproc_per_node, cfg, fine_tuning_dir, milestone)
    _run(
        [
            "tune",
            "run",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(nproc_per_node),
            *torchrun_flags,
            cfg.orpt_torchtune_recipe,
            "--config",
            f"torchtune_config/{cfg.orpt_torchtune_config}",
            *overrides,
            *extra_overrides,
        ],
        cwd=fine_tuning_dir,
        cfg=launch_cfg,
    )
    materialize_hf_checkpoint(ckpt_dir, cfg.orpt_epochs - 1, cfg.base_checkpoint_dir)
    if not (final_ckpt / "model.safetensors.index.json").exists():
        raise RuntimeError(f"Expected ORPT milestone {milestone} checkpoint at {final_ckpt}, but it wasn't produced")
    cleanup_intermediate_epochs(ckpt_dir, cfg.orpt_epochs - 1)
    return final_ckpt
