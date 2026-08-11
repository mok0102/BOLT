"""Shared low-level steps reused by both the trajectory chain and held-out
eval: sample+score candidates for one peptide task, and run one BO trial.
Kept separate from trajectory_chain.py/heldout_eval.py since both need the
exact same primitives, just pointed at different output directories.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from .config import ExperimentConfig

LOLBO_SCRIPTS_DIR = "optimization/peptides/lolbo_scripts"
FINE_TUNING_DIR = "fine-tuning/peptides"

# sampling_transformers.py generates `samples_per_peptide` sequences in a
# SINGLE model.generate(num_return_sequences=...) call, so memory scales
# with this number directly. Found via a real-scale (init_size=1000) sanity
# check: naively doubling this on every retry (1000 -> 16000) triggered a
# CUDA OOM on attempt 5. Cap it and accumulate multiple smaller calls
# instead of ever-larger single ones.
MAX_SAMPLES_PER_CALL = 2000


def _run(cmd: list, cwd: Path, cfg: ExperimentConfig | None = None) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"+ ({cwd}) {printable}", flush=True)
    # When this driver is itself attached to a real terminal (as opposed to a
    # redirected/nohup'd log file), the child inherits that tty on stdin/stdout.
    # Something in the LOLBO dependency chain then thinks it's interactive and
    # opens a pager (man/git/pydoc/rich all pick one up from PAGER/MANPAGER),
    # which blocks forever waiting for a keypress nobody will send -- observed
    # as a 13h+ hung run_lolbo child stuck in do_wait with ~0% CPU. Forcing
    # non-interactive pagers plus a closed stdin closes off every route to
    # that hang, regardless of which dependency triggers it.
    env = {**os.environ, "PAGER": "cat", "MANPAGER": "cat", "GIT_PAGER": "cat"}
    if cfg is not None and cfg.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd),
        check=True,
        env=env,
        stdin=subprocess.DEVNULL,
    )


def cleanup_intermediate_epochs(ckpt_dir: Path, final_ckpt: Path) -> None:
    """torchtune's lora_finetune_distributed/lora_dpo_distributed recipes save
    a full merged checkpoint after every epoch with no "final epoch only"
    option, so training for N epochs briefly leaves N full multi-GB
    checkpoints on disk. Delete every epoch_N dir except the final one once
    training is confirmed done. Shared by both the SFT (trajectory_chain.py)
    and ORPT (orpt.py) training steps.
    """
    for epoch_dir in ckpt_dir.glob("epoch_*"):
        if epoch_dir.is_dir() and epoch_dir != final_ckpt:
            shutil.rmtree(epoch_dir)


def _pad_to_size(init_path: Path, scores_path: Path, target_size: int) -> None:
    """Known failure mode (from prior smoke testing): an untuned/low-diversity
    model can plateau well below target_size unique candidates no matter how
    many retries. Pad by resampling with replacement rather than crashing.
    """
    seqs = [line for line in init_path.read_text().splitlines() if line.strip()]
    scores = [line for line in scores_path.read_text().splitlines() if line.strip()]
    n = min(len(seqs), len(scores))
    if n == 0:
        raise RuntimeError(f"No usable candidates produced in {init_path}")
    seqs, scores = seqs[:n], scores[:n]

    if n < target_size:
        rng = random.Random(0)
        pad_idx = [rng.randrange(n) for _ in range(target_size - n)]
        seqs = seqs + [seqs[i] for i in pad_idx]
        scores = scores + [scores[i] for i in pad_idx]
        print(f"  padded {n} -> {target_size} candidates by resampling with replacement")
    else:
        seqs, scores = seqs[:target_size], scores[:target_size]

    init_path.write_text("\n".join(seqs) + "\n")
    scores_path.write_text("\n".join(scores) + "\n")


def _ensure_constraint_feasible(
    cfg: ExperimentConfig, task_idx: int, init_path: Path, scores_path: Path
) -> None:
    """Found via smoke testing: an under-trained (or raw base) model can
    produce candidates that are ALL infeasible under the similarity
    constraint (none within cfg.similarity_threshold of the reference
    peptide). With zero feasible init points, LOLBO's trust-region logic
    spins in a near-infinite loop (rapid repeated 0% progress bars, no
    oracle-call progress) instead of erroring out. Guarantee a feasible
    floor by topping up with mutations of the reference sequence itself
    (the same method stbo_optimization.py uses), replacing the
    least-similar entries.
    """
    from Levenshtein import distance as edit_distance

    from apex_oracle import apex_wrapper
    from apex_oracle.init_data.create_mutations import generate_unique_mutations
    from apex_oracle.refseqs import REFERENCE_SEQUENCE

    reference_seq = REFERENCE_SEQUENCE[task_idx]
    seqs = [line for line in init_path.read_text().splitlines() if line.strip()]
    scores = [float(line) for line in scores_path.read_text().splitlines() if line.strip()]

    def similarity(seq: str) -> float:
        length = len(reference_seq)
        return (length - edit_distance(seq, reference_seq)) / length

    sims = [similarity(s) for s in seqs]
    n_feasible = sum(1 for s in sims if s >= cfg.similarity_threshold)
    min_feasible = max(1, len(seqs) // 10)
    if n_feasible >= min_feasible:
        return

    n_needed = min_feasible - n_feasible
    print(
        f"[task {task_idx}] only {n_feasible}/{len(seqs)} candidates satisfy the "
        f"similarity>={cfg.similarity_threshold} constraint; topping up {n_needed} "
        "with guaranteed-feasible mutations of the reference sequence"
    )
    mutations = generate_unique_mutations(
        [reference_seq],
        num_mutations=n_needed,
        max_mutation_distance=1.0 - cfg.similarity_threshold,
    )[0]
    mutation_scores = list(-apex_wrapper(mutations)[:, 0])

    # Replace the least-similar entries, keeping the most-similar existing ones.
    order = sorted(range(len(seqs)), key=lambda i: sims[i])
    for i, idx in enumerate(order[:n_needed]):
        seqs[idx] = mutations[i]
        scores[idx] = mutation_scores[i]

    init_path.write_text("\n".join(seqs) + "\n")
    scores_path.write_text("\n".join(f"{s:.8f}" for s in scores) + "\n")


def build_mutation_init(
    cfg: ExperimentConfig,
    task_idx: int,
    work_dir: Path,
) -> tuple[Path, Path]:
    """Build init candidates for a task with no fine-tuned checkpoint yet
    (i.e. before the first milestone) via mutations of the reference
    sequence, scored by the oracle -- the same method stbo_optimization.py
    uses, and the same method the original codebase used to precompute
    apex_oracle/init_data/seed_0_init.txt (see create_mutations.py).
    Sampling these tasks from the raw, untuned base model instead produces
    degenerate/low-diversity output (confirmed: see imp_plan progress log).
    """
    from apex_oracle import apex_wrapper
    from apex_oracle.init_data.create_mutations import generate_unique_mutations
    from apex_oracle.refseqs import REFERENCE_SEQUENCE

    work_dir.mkdir(parents=True, exist_ok=True)
    init_path = work_dir / f"task_{task_idx:04d}_init.txt"
    scores_path = work_dir / f"task_{task_idx:04d}_scores.csv"
    if init_path.exists() and scores_path.exists():
        print(f"[task {task_idx}] init data already exists at {init_path}, skipping")
        return init_path, scores_path

    reference_seq = REFERENCE_SEQUENCE[task_idx]
    mutations = generate_unique_mutations(
        [reference_seq],
        num_mutations=cfg.init_size,
        max_mutation_distance=1.0 - cfg.similarity_threshold,
    )[0]
    scores = list(-apex_wrapper(mutations)[:, 0])

    init_path.write_text("\n".join(mutations) + "\n")
    scores_path.write_text("\n".join(f"{s:.8f}" for s in scores) + "\n")
    _pad_to_size(init_path, scores_path, cfg.init_size)
    return init_path, scores_path


def sample_and_build_init(
    cfg: ExperimentConfig,
    model_path: Path | str,
    task_idx: int,
    work_dir: Path,
) -> tuple[Path, Path]:
    """Sample candidates for peptide task `task_idx` from `model_path`
    (a checkpoint dir, or the raw base model for task 0 / pre-milestone
    tasks), score them with the APEX oracle, and return (init_path,
    scores_path) ready to hand to the BO entry point.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    init_path = work_dir / f"task_{task_idx:04d}_init.txt"
    scores_path = work_dir / f"task_{task_idx:04d}_scores.csv"
    if init_path.exists() and scores_path.exists():
        print(f"[task {task_idx}] init data already exists at {init_path}, skipping")
        return init_path, scores_path

    fine_tuning_dir = cfg.bolt_root / FINE_TUNING_DIR

    pool_multiplier = 1
    max_attempts = 5
    n_unique = 0
    sample_jsonls: list[Path] = []
    for attempt in range(1, max_attempts + 1):
        samples_per_peptide = min(cfg.init_size * pool_multiplier, MAX_SAMPLES_PER_CALL)
        attempt_jsonl = work_dir / f"task_{task_idx:04d}_sampled_attempt{attempt}.jsonl"
        _run(
            [
                sys.executable,
                "sampling_transformers.py",
                "--model-path",
                model_path,
                "--start-index",
                task_idx,
                "--num-peptides",
                1,
                "--samples-per-peptide",
                samples_per_peptide,
                "--output-file",
                attempt_jsonl,
            ],
            cwd=fine_tuning_dir,
            cfg=cfg,
        )
        sample_jsonls.append(attempt_jsonl)
        _run(
            [
                sys.executable,
                "make_initialization_data.py",
                "--input-jsonl",
                *sample_jsonls,
                "--output-init",
                init_path,
                "--output-scores",
                scores_path,
                "--deduplicate",
            ],
            cwd=fine_tuning_dir / "sampled_output_from_ft",
            cfg=cfg,
        )
        n_unique = sum(1 for line in init_path.read_text().splitlines() if line.strip())
        if n_unique >= cfg.init_size:
            break
        print(
            f"[task {task_idx}] only {n_unique}/{cfg.init_size} unique candidates "
            f"after dedup across {len(sample_jsonls)} sampling call(s) "
            f"(attempt {attempt}/{max_attempts}), sampling {samples_per_peptide} more"
        )
        pool_multiplier *= 2
    else:
        print(
            f"[task {task_idx}] giving up on reaching {cfg.init_size} unique candidates "
            f"after {max_attempts} attempts ({n_unique} found); padding instead"
        )

    _pad_to_size(init_path, scores_path, cfg.init_size)
    _ensure_constraint_feasible(cfg, task_idx, init_path, scores_path)
    return init_path, scores_path


def run_bo(
    cfg: ExperimentConfig,
    task_idx: int,
    work_dir: Path,
    run_id: str,
    init_path: Path | None = None,
    scores_path: Path | None = None,
    stbo: bool = False,
    seed: int | None = None,
) -> Path:
    """Run one single-task BO trial (BOLT arm if init_path/scores_path are
    given, STBO arm if stbo=True) and return the path to its collected-data
    CSV (train_x, train_y), copied into `work_dir` for permanence.

    seed is an additive, opt-in knob (default None reproduces the exact
    subprocess CLI this function has always built): peptide_experiment/
    mi_orpt/one_step_evaluator.py::run_candidates_one_step needs a matched
    random seed shared across every candidate evaluated against the same
    background (paper/method.tex sec:one-step-pool-evaluation).
    Forwards directly to Optimize's own (already-existing, otherwise-unused)
    `--seed` constructor kwarg. Matched VAE initialization between the two
    arms needs no extra plumbing here: info_transformer_vae_optimization.py's
    own `path_to_vae_statedict` default already points both arms at the same
    fixed pretrained checkpoint unless overridden.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    dest_csv = work_dir / f"task_{task_idx:04d}.csv"
    if dest_csv.exists():
        print(f"[{run_id} task {task_idx}] trajectory already exists at {dest_csv}, skipping")
        return dest_csv

    lolbo_scripts_dir = cfg.bolt_root / LOLBO_SCRIPTS_DIR
    run_name = f"{run_id}_task_{task_idx:04d}"
    script = "stbo_optimization.py" if stbo else "info_transformer_vae_optimization.py"

    cmd = [
        sys.executable,
        script,
        "--task_id",
        "apex",
        "--max_n_oracle_calls",
        cfg.oracle_budget,
        "--bsz",
        cfg.bsz,
        "--constraint_function_ids",
        "[similarity]",
        "--constraint_thresholds",
        f"[{cfg.similarity_threshold}]",
        "--constraint_types",
        f"[{task_idx}]",
        "--num_initialization_points",
        cfg.init_size,
        "--init_n_update_epochs",
        20,
        "--max_string_length",
        30,
        "--task_specific_args",
        f"[{cfg.task_specific_args}]",
        "--init_offset_helper",
        task_idx,
        "--track_with_wandb",
        False,
        "--wandb_project_name",
        cfg.experiment_id,
        "--wandb_run_name",
        run_name,
    ]
    if not stbo:
        cmd += [
            "--init_data_path",
            init_path,
            "--init_scores_path",
            scores_path,
        ]
    if seed is not None:
        cmd += ["--seed", seed]
    cmd.append("run_lolbo")

    _run(cmd, cwd=lolbo_scripts_dir, cfg=cfg)

    produced_csv = (
        lolbo_scripts_dir
        / "optimization_all_collected_data"
        / f"{cfg.experiment_id}_{run_name}_all-data-collected.csv"
    )
    if not produced_csv.exists():
        raise RuntimeError(f"Expected BO output at {produced_csv}, but it wasn't produced")
    shutil.copy(produced_csv, dest_csv)
    return dest_csv
