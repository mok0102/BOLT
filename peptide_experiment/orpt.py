"""ORPT (Stage 4): a DPO preference-tuning stage trained on top of each
milestone's own BOLT-<m> SFT checkpoint. Mirrors trajectory_chain.py's
train_milestone() shape closely -- same cumulative-trajectory-data input,
same idempotent-skip-if-exists pattern -- but builds preference pairs
instead of a top-N SFT dataset, and trains via lora_dpo_distributed instead
of lora_finetune_distributed.

Called from trajectory_chain.py's run_trajectory_chain() immediately after
each train_milestone() call, only when cfg.build_orpt is True. See
imp_plan/01_peptide_reimplementation_plan.md and the ORPT section of the
durable plan for the full design rationale, in particular why ORPT-<m>
always branches fresh off that milestone's own BOLT-<m> rather than
chaining from ORPT-<m-1>, and why its output (not BOLT-<m>'s) becomes the
checkpoint that samples subsequent tasks.
"""

from __future__ import annotations

import dataclasses
import queue
import random
import sys
import threading

import yaml

from .config import ExperimentConfig
from .mi_orpt.candidate_bank import min_bank_size_needed
from .mi_orpt.warm_sampling_pool import create_sampling_pools, shutdown_sampling_pools
from .steps import _run, cleanup_intermediate_epochs, distributed_finetune_launch, materialize_hf_checkpoint, sample_and_build_init

FINE_TUNING_DIR = "fine-tuning/peptides"


def _sample_candidate_pool_parallel(cfg, model_path, task_indices, pool_dir, temperature):
    """Runs sample_and_build_init(cfg, model_path, i, pool_dir, temperature)
    for every i in task_indices, in parallel across cfg.mi_parallel_gpus if
    set (one worker thread per GPU, pulling tasks off a shared queue --
    mirrors trajectory_chain.py::_run_segment's design and its same
    documented in-process-CUDA-call caveat), else serially in order. Returns
    {task_idx: (init_path, scores_path)}.
    """
    results: dict = {}
    gpus = cfg.mi_parallel_gpus
    if not gpus:
        for i in task_indices:
            results[i] = sample_and_build_init(cfg, model_path, i, pool_dir, temperature=temperature)
        return results

    work_queue: queue.Queue = queue.Queue()
    for i in task_indices:
        work_queue.put(i)
    results_lock = threading.Lock()
    errors: list = []
    stop_event = threading.Event()

    # One resident copy of this milestone's BOLT-<m> per GPU instead of a fresh
    # checkpoint load per task (see mi_orpt/warm_sampling_pool.py). model_path
    # is a single checkpoint by construction here, so unlike trajectory_chain's
    # _run_segment there is no cross-task invariant left to check.
    pools = create_sampling_pools(gpus, cfg.bolt_root / FINE_TUNING_DIR, model_path)

    def worker(gpu: str) -> None:
        task_cfg = dataclasses.replace(cfg, cuda_visible_devices=gpu)
        sampling_pool = pools.get(gpu)
        while not stop_event.is_set():
            try:
                i = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                result = sample_and_build_init(
                    task_cfg, model_path, i, pool_dir, temperature=temperature, sampling_pool=sampling_pool
                )
            except Exception as e:  # noqa: BLE001 -- collected, re-raised below, not swallowed
                with results_lock:
                    errors.append((i, e))
                stop_event.set()
                return
            with results_lock:
                results[i] = result

    try:
        threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            task_idx, exc = errors[0]
            raise RuntimeError(f"mi_candidate_temperature sampling for task {task_idx} failed: {exc!r}") from exc
        return results
    finally:
        # Scoped to this function on purpose: build_orpt_pairs goes straight
        # from here into the build_pairs.py subprocess, which stands up its OWN
        # warm_pool + warm_scoring_pool on these same GPUs, and train_orpt_
        # milestone then runs DPO at ~36GB/GPU. Hoisting the pool any higher
        # would stack resident models on top of both.
        shutdown_sampling_pools(pools)


def build_orpt_pairs(cfg: ExperimentConfig, milestone: int):
    """Build preference pairs from the same cumulative trajectory data
    [0, milestone) that BOLT-<milestone>'s own SFT dataset uses.

    Dispatches on cfg.orpt_pair_source (imp_plan/06_orpt_matched_intervention_plan.md):
    "objective_ranked" (default) uses the original codebase's own
    uniform-random-pairing strategy (verified byte-identical against
    /workspace/mok/BOLT's make_dpo_train_data_csv.py for
    cfg.orpt_pairing_mode == "feasible_only", its default) -- restricts to
    constraint-feasible candidates and ranks purely by objective score (see
    that script's sample_pairs()/_pick_chosen_rejected()).
    "matched_intervention" instead labels pairs via a matched one-candidate
    intervention evaluated with actual one-step BO (paper/method.tex;
    peptide_experiment/mi_orpt/) -- real oracle calls during pair
    construction, per cfg.mi_bo_steps (1 = one-step BO, 0 = the zero-step
    ablation with no additional oracle cost). pi_ref for this is plain
    BOLT-<milestone> itself, since no ORPT LoRA adapter exists yet for this
    milestone.
    """
    pairs_csv = cfg.orpt_pairs_dir / f"orpt_pairs_{milestone}.csv"
    pairs_jsonl = cfg.orpt_pairs_dir / f"orpt_pairs_{milestone}.jsonl"

    if pairs_jsonl.exists():
        print(f"[orpt pairs {milestone}] already exists at {pairs_jsonl}, skipping")
        return pairs_jsonl

    task_indices = list(range(milestone))
    if (
        cfg.orpt_pair_source == "matched_intervention"
        and cfg.mi_max_tasks_per_milestone is not None
        and len(task_indices) > cfg.mi_max_tasks_per_milestone
    ):
        # Fixed-seed uniform subsample, not the first N -- pair construction
        # cost is linear in len(task_indices) (see mi_max_tasks_per_milestone's
        # own docstring), so at the largest milestones this is the difference
        # between hundreds and thousands of one-step-BO subprocess calls.
        # BOLT-<milestone>'s own SFT dataset (train_milestone(), a sibling
        # call) still uses every task in [0, milestone) -- only the DPO
        # stage's pair pool is subsampled. Same seed every time this
        # milestone is (re-)built, for reproducibility.
        task_indices = sorted(random.Random(1000 + milestone).sample(task_indices, cfg.mi_max_tasks_per_milestone))
        print(
            f"[orpt pairs {milestone}] mi_max_tasks_per_milestone={cfg.mi_max_tasks_per_milestone}: "
            f"using {len(task_indices)}/{milestone} tasks for pair construction"
        )
    input_csvs = [cfg.trajectories_csv_dir / f"task_{i:04d}.csv" for i in task_indices]
    reference_indices = [str(i) for i in task_indices]

    if cfg.orpt_pair_source == "matched_intervention":
        torchtune_config_path = cfg.bolt_root / FINE_TUNING_DIR / "torchtune_config" / cfg.torchtune_config

        candidate_inits: list = []
        candidate_scores: list = []
        if cfg.mi_candidate_temperature is not None:
            # Dedicated candidate pool for mi_orpt's own bank, sampled at
            # cfg.mi_candidate_temperature -- separate from trajectories_csv_dir,
            # so BOLT-<milestone>'s own SFT dataset (built from trajectories_csv_dir
            # elsewhere) is unaffected by this temperature choice.
            # Milestone-scoped: this pool is sampled from BOLT-<milestone>, which
            # is a DIFFERENT model at every milestone, but sample_and_build_init
            # skips whenever task_<i>_init.txt already exists. An unscoped path
            # therefore makes milestone 20+ silently reuse milestone 10's pool
            # (sampled from the wrong, older checkpoint) instead of resampling
            # -- same class of bug as mi_onestep_bo's own milestone_XXXX/ fix.
            pool_dir = cfg.run_dir / "mi_candidate_pool" / f"milestone_{milestone:04d}"
            # Target unique-candidate count, NOT cfg.init_size: sample_and_build_init's
            # own retry loop only chases cfg.init_size unique draws (pre-feasibility-
            # filter), which is mathematically guaranteed to fall short of
            # min_bank_size_needed(init_size, mi_max_candidates_per_task) once
            # infeasible/duplicate draws are filtered out downstream -- see
            # cfg.mi_candidate_pool_size's docstring.
            #
            # The multiplier is load-bearing, not a safety fudge: steps.py::
            # _ensure_constraint_feasible only guarantees a feasibility FLOOR of
            # len(pool)//10 (`min_feasible = max(1, len(seqs) // 10)`), so a pool
            # of size P yields at least P//10 similarity-feasible candidates and,
            # empirically at milestone 10, not many more (natural feasible rate
            # measured at ~5-11% -- the floor is usually what binds). The bank
            # needs min_bank_size_needed(...) feasible candidates, so the pool
            # must be >= 10x that or the bank gate can never be met. The previous
            # x2 predated that floor and silently guaranteed only
            # (2*115)//10 = 23 feasible against the 115 needed -- i.e. it could
            # only ever work if the natural feasible rate exceeded 50%, which
            # cfg.mi_candidate_pool_size's own "57-78%" note assumed but which
            # this run's real checkpoints do not come close to.
            #
            # x15 rather than a bare x10 because the floor counts feasible
            # entries WITH duplicates (_pad_to_size pads by resampling with
            # replacement) while the bank dedups by sequence, so exactly-10x
            # can still land under the bar once duplicates collapse.
            pool_size = cfg.mi_candidate_pool_size or (
                min_bank_size_needed(cfg.mi_background_size or cfg.init_size, cfg.mi_max_candidates_per_task) * 15
            )
            pool_cfg = dataclasses.replace(cfg, init_size=pool_size)
            results = _sample_candidate_pool_parallel(
                pool_cfg, cfg.milestone_checkpoint_dir(milestone), task_indices, pool_dir, cfg.mi_candidate_temperature
            )
            for i in task_indices:
                init_path, scores_path = results[i]
                candidate_inits.append(init_path)
                candidate_scores.append(scores_path)

        cmd = [
            sys.executable,
            "-m",
            "peptide_experiment.mi_orpt.build_pairs",
            "--input-csv",
            *input_csvs,
            "--reference-index",
            *reference_indices,
            "--output-csv",
            pairs_csv,
            "--output-jsonl",
            pairs_jsonl,
            "--torchtune-config",
            torchtune_config_path,
            "--checkpoint-dir",
            cfg.milestone_checkpoint_dir(milestone),
            "--similarity-threshold",
            cfg.similarity_threshold,
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
            "--task-specific-args",
            cfg.task_specific_args,
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
            cmd += ["--parallel-gpus", *cfg.mi_parallel_gpus]
            # CUDA_VISIBLE_DEVICES only narrows going down a process tree --
            # this build_pairs.py subprocess (and the LOLBO grandchildren it
            # spawns) must see every GPU in the pool up front so each
            # grandchild's own narrower per-call override can still resolve
            # to the right physical device (mi_orpt/one_step_evaluator.py::
            # run_candidates_one_step). A single cuda_visible_devices pin
            # here would otherwise silently collapse every parallel call
            # onto just that one GPU.
            launch_cfg = dataclasses.replace(cfg, cuda_visible_devices=",".join(cfg.mi_parallel_gpus))
        _run(cmd, cwd=cfg.bolt_root, cfg=launch_cfg)
        return pairs_jsonl

    assert cfg.orpt_pair_source == "objective_ranked", cfg.orpt_pair_source
    fine_tuning_dir = cfg.bolt_root / FINE_TUNING_DIR
    _run(
        [
            sys.executable,
            "make_dpo_train_data_csv.py",
            "--input-csv",
            *input_csvs,
            "--reference-index",
            *reference_indices,
            "--output-csv",
            pairs_csv,
            "--output-jsonl",
            pairs_jsonl,
            "--similarity-threshold",
            cfg.similarity_threshold,
            "--pairs-per-input",
            cfg.orpt_pairs_per_task,
            "--pairing-mode",
            cfg.orpt_pairing_mode,
            "--seed",
            42,
        ],
        cwd=fine_tuning_dir,
        cfg=cfg,
    )
    return pairs_jsonl


def build_infeasible_singles(cfg: ExperimentConfig, milestone: int):
    """Only used when cfg.orpt_loss_type == "dpo_ii_penalty": builds a JSONL
    of individually-sampled (not paired) similarity-infeasible completions
    from the same cumulative trajectory data [0, milestone) build_orpt_pairs
    uses, via make_infeasible_singles_csv.py -- see that script's docstring
    for why no pairing is needed for this loss's infeasible-suppression term.
    """
    fine_tuning_dir = cfg.bolt_root / FINE_TUNING_DIR
    singles_jsonl = cfg.orpt_pairs_dir / f"orpt_infeasible_singles_{milestone}.jsonl"

    if singles_jsonl.exists():
        print(f"[orpt infeasible singles {milestone}] already exists at {singles_jsonl}, skipping")
        return singles_jsonl

    input_csvs = [cfg.trajectories_csv_dir / f"task_{i:04d}.csv" for i in range(milestone)]
    reference_indices = [str(i) for i in range(milestone)]

    _run(
        [
            sys.executable,
            "make_infeasible_singles_csv.py",
            "--input-csv",
            *input_csvs,
            "--reference-index",
            *reference_indices,
            "--output-jsonl",
            singles_jsonl,
            "--similarity-threshold",
            cfg.similarity_threshold,
            "--singles-per-input",
            cfg.orpt_infeasible_singles_per_task,
            "--seed",
            42,
        ],
        cwd=fine_tuning_dir,
        cfg=cfg,
    )
    return singles_jsonl


MIN_DPO_STEPS = 8


def _fit_dpo_batch_size(extra_overrides, pairs_jsonl, nproc_per_node, cfg, fine_tuning_dir, milestone):
    """Shrink the DPO per-device batch_size until this milestone's pair count
    actually yields optimizer steps, and refuse to launch if it can't.

    torchtune's distributed recipes drop the last incomplete batch, so a
    dataset smaller than batch_size*world_size produces ZERO batches -- the
    recipe still runs, still saves a checkpoint, and still reports success,
    but no optimizer step ever happens. Because LoRA's B matrix is
    zero-initialised, that checkpoint is numerically IDENTICAL to the
    BOLT-<m> it started from: an ORPT arm that silently degenerates into a
    copy of the SFT arm, which would invalidate the whole comparison rather
    than fail loudly.

    Hit for real at milestone 10: 70 pairs against batch_size=32 on 4 GPUs
    (global batch 128) -> `0it [03:03]`, no Loss lines, ORPT-10 == BOLT-10.
    The YAML's 32 is tuned for the SFT-scale datasets (thousands of rows);
    matched-intervention pair counts are one to two orders of magnitude
    smaller (~10/task, capped by mi_max_tasks_per_milestone), and they vary
    per milestone, so this has to be derived from the data rather than
    pinned in config. Halving (rather than picking an exact divisor) keeps
    the batch a power-of-two multiple of the tuned value's memory profile.
    """
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


def train_orpt_milestone(cfg: ExperimentConfig, milestone: int):
    """Train ORPT-<milestone>: a DPO stage on top of that same milestone's
    own BOLT-<milestone> checkpoint (pi_theta init == pi_ref, via
    lora_dpo_distributed's LoRA adapter-disable trick -- no separate
    ref_checkpointer needed).
    """
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
    pairs_jsonl = build_orpt_pairs(cfg, milestone)
    if pairs_jsonl.stat().st_size == 0:
        # Only reachable via orpt_pair_source="matched_intervention" --
        # "objective_ranked"'s sample_pairs() always either reaches
        # orpt_pairs_per_task or raises, so it can never produce an empty
        # file. matched_intervention's reliability filter can legitimately
        # yield zero pairs for every task in a milestone (paper/appendix.tex
        # app:pair-construction's reliability criterion) -- surface that
        # clearly instead of letting torchtune's DPO recipe fail on an
        # empty dataset with an opaque ChildFailedError.
        raise RuntimeError(
            f"[orpt milestone {milestone}] {pairs_jsonl} has 0 preference pairs "
            f"(orpt_pair_source={cfg.orpt_pair_source!r}) -- nothing to train ORPT-{milestone} on. "
            "Likely cause: eligible bank too small at this milestone/scale for "
            "min_bank_size_needed(m)=m+1 across every task. Not auto-skipped: "
            "downstream checkpoint_to_sample_from() has no fallback for a missing "
            "ORPT-<m> when build_orpt=True, so silently continuing would only move "
            "the failure later and obscure the cause."
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
    if cfg.orpt_loss_type == "dpo_ii_penalty":
        # loss.beta (already appended above) drives the untouched stock
        # DPOLoss; dpo_ii's InfeasibleSuppressionLoss (fine-tuning/peptides/
        # dpo_ii/loss.py) additionally takes gamma_i/lambda_inf, and its
        # dataset needs the individually-sampled infeasible-singles JSONL
        # (see build_infeasible_singles above).
        singles_jsonl = build_infeasible_singles(cfg, milestone)
        overrides += [
            f"ii_dataset.data_files={singles_jsonl}",
            f"ii_loss.gamma_i={cfg.fa_orpt_gamma_i}",
            f"ii_loss.lambda_inf={cfg.fa_orpt_lambda_inf}",
        ]
    elif cfg.orpt_loss_type == "bpo":
        # bpo_loss.BPOLoss is a drop-in loss._component_ replacement under the
        # stock lora_dpo_distributed recipe (see fine-tuning/peptides/bpo_loss.py) --
        # loss.beta (already appended above) is reused as-is; only the extra
        # alpha "gap adaptor" hyperparameter needs threading through.
        overrides += [f"loss.alpha={cfg.orpt_bpo_alpha}"]

    torchrun_flags, extra_overrides, launch_cfg = distributed_finetune_launch(cfg, cfg.orpt_torchtune_config, fine_tuning_dir)
    nproc_per_node = len(cfg.mi_parallel_gpus) if cfg.mi_parallel_gpus else 1
    extra_overrides = _fit_dpo_batch_size(
        extra_overrides, pairs_jsonl, nproc_per_node, cfg, fine_tuning_dir, milestone
    )
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
        raise RuntimeError(
            f"Expected ORPT milestone {milestone} checkpoint at {final_ckpt}, but it wasn't produced"
        )
    cleanup_intermediate_epochs(ckpt_dir, cfg.orpt_epochs - 1)
    return final_ckpt
