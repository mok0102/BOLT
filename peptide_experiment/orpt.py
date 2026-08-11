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
import sys

from .config import ExperimentConfig
from .steps import _run, cleanup_intermediate_epochs

FINE_TUNING_DIR = "fine-tuning/peptides"


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

    input_csvs = [cfg.trajectories_csv_dir / f"task_{i:04d}.csv" for i in range(milestone)]
    reference_indices = [str(i) for i in range(milestone)]

    if cfg.orpt_pair_source == "matched_intervention":
        torchtune_config_path = cfg.bolt_root / FINE_TUNING_DIR / "torchtune_config" / cfg.torchtune_config
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
            cfg.init_size,
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
            "--experiment-id",
            cfg.experiment_id,
            "--bsz",
            cfg.bsz,
            "--task-specific-args",
            cfg.task_specific_args,
            "--seed",
            42,
        ]
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


def train_orpt_milestone(cfg: ExperimentConfig, milestone: int):
    """Train ORPT-<milestone>: a DPO stage on top of that same milestone's
    own BOLT-<milestone> checkpoint (pi_theta init == pi_ref, via
    lora_dpo_distributed's LoRA adapter-disable trick -- no separate
    ref_checkpointer needed).
    """
    ckpt_dir = cfg.checkpoints_dir / f"ORPT-{milestone}"
    final_ckpt = cfg.orpt_checkpoint_dir(milestone)
    if final_ckpt.exists():
        print(f"[orpt milestone {milestone}] checkpoint already exists at {final_ckpt}, skipping DPO")
        return final_ckpt

    bolt_ckpt = cfg.milestone_checkpoint_dir(milestone)
    if not bolt_ckpt.exists():
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

    _run(
        [
            "tune",
            "run",
            "--nnodes",
            "1",
            "--nproc_per_node",
            "1",
            cfg.orpt_torchtune_recipe,
            "--config",
            f"torchtune_config/{cfg.orpt_torchtune_config}",
            *overrides,
        ],
        cwd=fine_tuning_dir,
        cfg=cfg,
    )
    if not final_ckpt.exists():
        raise RuntimeError(
            f"Expected ORPT milestone {milestone} checkpoint at {final_ckpt}, but it wasn't produced"
        )
    cleanup_intermediate_epochs(ckpt_dir, final_ckpt)
    return final_ckpt
