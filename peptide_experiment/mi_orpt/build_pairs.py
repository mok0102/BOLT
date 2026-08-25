"""CLI driver for matched-intervention preference-pair construction
(paper/method.tex sec:one-step-pool-evaluation). Mirrors
fine-tuning/peptides/make_dpo_train_data_csv.py's own
--input-csv/--reference-index/--output-jsonl invocation shape (same
multi-task-per-call convention peptide_experiment/orpt.py::build_orpt_pairs
already uses) so it drops into the same call site; emits the identical
{"chosen": [...], "rejected": [...]} JSONL shape via
make_dpo_train_data_csv.py's own make_messages()/write_jsonl(), so the
existing, unmodified torchtune DPO recipe consumes it unchanged.

With --bo-steps 1 (the paper's actual method) this spawns real run_bo
subprocesses -- new oracle calls during pair construction, per
appendix.tex's "Training-Time Oracle Evaluation" -- not a bug, the method's
documented cost. --bo-steps 0 is the zero-step ablation (experiments.tex
sec:ablations), which makes no additional oracle calls at all.

Usage (run from the BOLT repo root, matching build_orpt_pairs' cwd):
    python -m peptide_experiment.mi_orpt.build_pairs \\
        --input-csv runs/<id>/trajectories_csv/task_0000.csv \\
        --reference-index 0 \\
        --output-csv pairs.csv --output-jsonl pairs.jsonl \\
        --torchtune-config qwen_2_5_3B_lora.yaml --checkpoint-dir BOLT-2 \\
        --experiment-id <id> --bsz 50 --oracle-budget 5000 \\
        --m 500 --target-pairs-per-task 5 --max-candidates-per-task 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

FINE_TUNING_DIR = Path(__file__).resolve().parents[2] / "fine-tuning" / "peptides"
if str(FINE_TUNING_DIR) not in sys.path:
    sys.path.insert(0, str(FINE_TUNING_DIR))

from make_dpo_train_data_csv import SYSTEM_PROMPT, load_reference_sequence, make_messages  # noqa: E402

from ..config import ExperimentConfig
from .candidate_bank import build_eligible_bank, build_eligible_bank_from_init_scores, min_bank_size_needed
from .pair_construction import construct_pairs_for_task
from .warm_pool import create_pool
from .warm_scoring_pool import create_scoring_pool, score_sequences_parallel

PAIRS_CSV_FIELDNAMES = ["reference_sequence", "chosen_sequence", "rejected_sequence", "chosen_score", "rejected_score", "delta", "se"]


def write_jsonl(pairs: list[dict], output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as f_out:
        for pair in pairs:
            messages = {
                "chosen": make_messages(pair["reference_sequence"], pair["chosen_sequence"]),
                "rejected": make_messages(pair["reference_sequence"], pair["rejected_sequence"]),
            }
            f_out.write(json.dumps(messages) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-index", type=int, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--torchtune-config", type=Path, required=True, help="pi_ref's SFT config (cfg.torchtune_config)")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="pi_ref checkpoint (BOLT-<milestone>)")
    parser.add_argument("--similarity-threshold", type=float, default=0.75)
    parser.add_argument("--m", type=int, required=True, help="pool size -- reuses cfg.init_size")
    parser.add_argument("--target-pairs-per-task", type=int, required=True, help="stop once this many reliable pairs are found for a task")
    parser.add_argument("--max-candidates-per-task", type=int, required=True, help="give up after evaluating this many candidates, even short of target -- each candidate costs exactly --num-backgrounds real-BO calls")
    parser.add_argument("--num-backgrounds", type=int, default=8, help="M")
    parser.add_argument("--tau-q", type=float, default=1.0)
    parser.add_argument("--z-min", type=float, default=1.96)
    parser.add_argument("--delta-t", type=float, default=0.0)
    parser.add_argument("--bo-steps", type=int, choices=[0, 1], default=1, help="1 = actual one-step BO (method.tex); 0 = zero-step ablation (experiments.tex)")
    parser.add_argument("--experiment-id", required=True, help="reused as the one-step BO subprocess's wandb_project_name/work-dir namespace")
    parser.add_argument("--bsz", type=int, required=True, help="deployment acquisition batch size -- one-step's oracle_budget")
    parser.add_argument("--task-specific-args", default="bacteria_0")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--parallel-gpus", nargs="*", default=None, help="CUDA device ids to round-robin across for concurrent one-step BO calls -- omit for serial (default)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidate-init",
        type=Path,
        nargs="+",
        default=None,
        help="Per-task init.txt from a dedicated cfg.mi_candidate_temperature sampling pass "
        "(steps.py::sample_and_build_init), one per --input-csv in the same order. When given "
        "(together with --candidate-scores), the eligible bank is built from these instead of "
        "--input-csv's trajectory CSV -- --input-csv/--reference-index are still required (only "
        "the bank-building source changes, not task selection).",
    )
    parser.add_argument(
        "--candidate-scores",
        type=Path,
        nargs="+",
        default=None,
        help="Per-task scores.csv paired 1:1 with --candidate-init.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    # Minimal ExperimentConfig reconstruction: run_one_step_pool()/run_bo()
    # only ever reads oracle_budget/bsz/similarity_threshold/
    # task_specific_args/experiment_id/cuda_visible_devices/bolt_root from
    # it (oracle_budget/init_size get overridden per-pool anyway) --
    # milestones/sft_epochs are irrelevant here since pi_ref's checkpoint is
    # already resolved (--checkpoint-dir) and no SFT/ORPT training happens
    # in this process.
    cfg = ExperimentConfig(
        experiment_id=args.experiment_id,
        milestones=[1],
        oracle_budget=args.bsz,
        init_size=args.m,
        bsz=args.bsz,
        sft_epochs=1,
        similarity_threshold=args.similarity_threshold,
        task_specific_args=args.task_specific_args,
        cuda_visible_devices=args.cuda_visible_devices,
        mi_parallel_gpus=args.parallel_gpus,
    )

    reference_sequences = [load_reference_sequence(i) for i in args.reference_index]
    if len(args.input_csv) != len(reference_sequences):
        raise ValueError(f"Expected one reference per input CSV, got {len(args.input_csv)} CSVs and {len(reference_sequences)} references.")

    if args.candidate_init is not None or args.candidate_scores is not None:
        if args.candidate_init is None or args.candidate_scores is None:
            raise ValueError("--candidate-init and --candidate-scores must be given together")
        if len(args.candidate_init) != len(args.input_csv) or len(args.candidate_scores) != len(args.input_csv):
            raise ValueError(
                f"Expected one --candidate-init/--candidate-scores pair per --input-csv, got "
                f"{len(args.candidate_init)} candidate-init, {len(args.candidate_scores)} candidate-scores, "
                f"{len(args.input_csv)} input-csv."
            )
    else:
        args.candidate_init = [None] * len(args.input_csv)
        args.candidate_scores = [None] * len(args.input_csv)

    # Created once, shared across every task in this milestone (not
    # recreated per task) -- the whole point of the warm pool is
    # amortizing the ~11s LOLBO import cost over every one-step-BO call
    # made during this entire build_pairs.py invocation, not just one
    # task's worth (mi_orpt/warm_pool.py).
    worker_pool = create_pool(args.parallel_gpus, cfg.bolt_root) if args.parallel_gpus else None
    # Same rationale, applied to reference-model scoring (likelihood.py):
    # every task in this invocation scores against the exact same milestone
    # checkpoint, so the model loads once here -- either once per GPU in a
    # parallel scoring pool (chunked across GPUs per task), or once total
    # in the serial fallback -- never once per task (measured ~3.25s wasted
    # per task previously, plus scoring itself was 100% serialized on a
    # single GPU regardless of mi_parallel_gpus; see mi_orpt/warm_scoring_pool.py).
    scoring_pool = create_scoring_pool(args.parallel_gpus, args.torchtune_config, args.checkpoint_dir) if args.parallel_gpus else None
    scoring_model = scoring_tokenizer = None
    if scoring_pool is None:
        # Lazy on purpose: likelihood.py imports torchtune, which
        # eagerly initializes a CUDA context on the *default* device
        # (physical GPU matching whatever index 0 resolves to) just by
        # being imported -- see warm_pool.py's own docstring for why this
        # class of bug is dangerous specifically at this module's top
        # level. Every worker warm_pool.py/warm_scoring_pool.py spawns
        # re-executes build_pairs.py's top-level imports (multiprocessing
        # "spawn" + "-m" entry-point re-import), *before* that worker's
        # own per-GPU CUDA_VISIBLE_DEVICES narrowing runs -- a top-level
        # import here would silently pin every worker's CUDA context to
        # the wide, unnarrowed device view (physical GPU 0 as the
        # default), making the later narrowing a no-op and collapsing
        # every worker onto the same physical GPU (confirmed via a real
        # OOM: 8 one-step-BO workers all landing on GPU 0). Deferring the
        # import to here means it only ever runs in the main process,
        # after cuda_visible_devices/parallel-gpus have already been
        # decided, and only when the serial (no --parallel-gpus) fallback
        # path is actually taken.
        from .likelihood import load_model_and_tokenizer, score_sequences

        scoring_model, scoring_tokenizer = load_model_and_tokenizer(args.torchtune_config, args.checkpoint_dir)
    try:
        all_pairs: list[dict] = []
        for input_csv, task_idx, reference_sequence, candidate_init, candidate_scores in zip(
            args.input_csv, args.reference_index, reference_sequences, args.candidate_init, args.candidate_scores
        ):
            if candidate_init is not None:
                bank = build_eligible_bank_from_init_scores(candidate_init, candidate_scores, reference_sequence, args.similarity_threshold)
            else:
                bank = build_eligible_bank(input_csv, reference_sequence, args.similarity_threshold)
            min_needed = min_bank_size_needed(args.m, num_reserved=args.max_candidates_per_task)
            if len(bank) < min_needed:
                print(f"[build_pairs] {input_csv}: bank has {len(bank)} candidates, need >= {min_needed}, skipping task")
                continue

            if scoring_pool is not None:
                log_probs = score_sequences_parallel(
                    scoring_pool,
                    len(args.parallel_gpus),
                    context=reference_sequence,
                    sequences=[c.seq for c in bank],
                    system_prompt=SYSTEM_PROMPT,
                )
            else:
                log_probs = score_sequences(
                    scoring_model,
                    scoring_tokenizer,
                    context=reference_sequence,
                    sequences=[c.seq for c in bank],
                    system_prompt=SYSTEM_PROMPT,
                )
            log_likelihoods = {c.seq: lp for c, lp in zip(bank, log_probs)}

            work_dir_root = cfg.orpt_pairs_dir / "mi_onestep_bo" / f"task_{task_idx:04d}"
            task_pairs = construct_pairs_for_task(
                cfg=cfg,
                task_idx=task_idx,
                bank=bank,
                reference_sequence=reference_sequence,
                log_likelihoods=log_likelihoods,
                m=args.m,
                target_pairs=args.target_pairs_per_task,
                max_candidates=args.max_candidates_per_task,
                num_backgrounds=args.num_backgrounds,
                tau_q=args.tau_q,
                z_min=args.z_min,
                delta_t=args.delta_t,
                bo_steps=args.bo_steps,
                work_dir_root=work_dir_root,
                rng=rng,
                worker_pool=worker_pool,
            )
            all_pairs.extend(task_pairs)
    finally:
        if worker_pool is not None:
            worker_pool.shutdown(wait=True)
        if scoring_pool is not None:
            scoring_pool.shutdown(wait=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=PAIRS_CSV_FIELDNAMES)
        writer.writeheader()
        for pair in all_pairs:
            writer.writerow({k: pair[k] for k in PAIRS_CSV_FIELDNAMES})

    write_jsonl(all_pairs, args.output_jsonl)
    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote JSONL: {args.output_jsonl}")
    print(f"Total matched-intervention DPO pairs: {len(all_pairs)}")


if __name__ == "__main__":
    main()
