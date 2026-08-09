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
        --m 500 --target-pairs-per-task 5 --max-pair-attempts-per-task 20
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
from .candidate_bank import build_eligible_bank
from .likelihood import score_sequences
from .pair_construction import construct_pairs_for_task
from .warm_pool import create_pool

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
    parser.add_argument("--max-pair-attempts-per-task", type=int, required=True, help="give up after this many candidate pairs tried, even short of target")
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

    # Created once, shared across every task in this milestone (not
    # recreated per task) -- the whole point of the warm pool is
    # amortizing the ~11s LOLBO import cost over every one-step-BO call
    # made during this entire build_pairs.py invocation, not just one
    # task's worth (mi_orpt/warm_pool.py).
    worker_pool = create_pool(args.parallel_gpus, cfg.bolt_root) if args.parallel_gpus else None
    try:
        all_pairs: list[dict] = []
        for input_csv, task_idx, reference_sequence in zip(args.input_csv, args.reference_index, reference_sequences):
            bank = build_eligible_bank(input_csv, reference_sequence, args.similarity_threshold)
            if len(bank) < args.m + 1:
                print(f"[build_pairs] {input_csv}: bank has {len(bank)} candidates, need >= {args.m + 1}, skipping task")
                continue

            log_probs = score_sequences(
                torchtune_config_path=args.torchtune_config,
                checkpoint_dir=args.checkpoint_dir,
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
                max_attempts=args.max_pair_attempts_per_task,
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
