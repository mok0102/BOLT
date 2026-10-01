"""CLI driver for matched-intervention preference-pair construction (mol
domain). Mirrors peptide_experiment/mi_orpt/build_pairs.py's own
--input-csv/--task-index/--output-jsonl invocation shape (read-only
reference, never imported -- isolation contract) so it drops into
mol_experiment/orpt.py's analogous call site the same way; emits the
identical {"chosen": [...], "rejected": [...]} JSONL shape via
optimization/mol/mol_prompt.py's own make_messages(), so the existing,
unmodified torchtune DPO recipe consumes it unchanged.

Task identity differs from peptide's: mol has no single "reference
sequence" string. --task-index resolves to optimization/mol/mol_tasks.py's
MolTask (target protein sequence + seed SELFIES + seed SMILES), so this
file takes --task-index directly rather than peptide's --reference-index +
load_reference_sequence().

With --bo-steps 1 (the paper's actual method) this spawns real mol_run_bo
calls -- new oracle calls during pair construction, the method's
documented cost. --bo-steps 0 is the zero-step ablation, which makes no
additional oracle calls at all.

--parallel-gpus dispatches across TWO independent pools built from the same
GPU list, mirroring peptide's own build_pairs.py: reference-model scoring
(mi_orpt/warm_scoring_pool.py) and one-step-BO evaluation
(mi_orpt/warm_pool.py) -- kept as separate pools deliberately (different
loaded state, different lifetime granularity) rather than one pool doing
double duty, same reasoning as warm_scoring_pool.py's own docstring. Without
--parallel-gpus, both fall back to serial, on this process's own device.

Usage (run from the BOLT repo root):
    python -m mol_experiment.mi_orpt.build_pairs \\
        --input-csv runs/mol/<id>/trajectories_csv/task_0000.csv \\
        --task-index 0 \\
        --output-csv pairs.csv --output-jsonl pairs.jsonl \\
        --torchtune-config fine-tuning/mol/torchtune_config/qwen_2_5_3B_lora.yaml \\
        --checkpoint-dir runs/mol/<id>/checkpoints/BOLT-2 \\
        --experiment-id <id> --bsz 5 --milestone 2 \\
        --m 20 --target-pairs-per-task 5 --max-candidates-per-task 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mol_experiment.config import MolExperimentConfig  # noqa: E402
from mol_prompt import SYSTEM_PROMPT, make_messages, render_context  # noqa: E402
from mol_tasks import MolTask, get_task  # noqa: E402

from .candidate_bank import build_eligible_bank, build_eligible_bank_from_init_scores  # noqa: E402
from .pair_construction import construct_pairs_for_task  # noqa: E402
from .warm_pool import create_mol_pool  # noqa: E402
from .warm_scoring_pool import create_scoring_pool, score_sequences_parallel  # noqa: E402

PAIRS_CSV_FIELDNAMES = ["task_context", "chosen_sequence", "rejected_sequence", "chosen_score", "rejected_score", "delta", "se"]

_VOCAB = None


def smiles_to_selfies_vocab_checked(smiles: str) -> str | None:
    """mi_orpt's candidates (EligibleCandidate.seq, chosen_sequence/
    rejected_sequence) are canonical SMILES -- candidate_bank.py's own
    docstring, inherited from mol_run_bo's trajectory CSV -- but
    mol_prompt.py's system prompt requires the LLM's assistant turn to be a
    SELFIES string ("Output only the candidate SELFIES string"). This is the
    one conversion point between the two representations, same
    selfies.encoder + split_selfies + vocab-membership pattern as
    fine-tuning/mol/make_train_data_csv.py's own copy (kept separate rather
    than shared: that script lives in a different tree and mi_orpt already
    tolerates small per-file feasibility-check duplication, e.g.
    candidate_bank.py's own is_similar_enough)."""
    global _VOCAB
    if _VOCAB is None:
        from mol_lolbo.utils.mol_utils.selfies_vae.model_positional_unbounded import SELFIESDataset

        _VOCAB = SELFIESDataset().vocab2idx
    import selfies as sf

    try:
        selfies_str = sf.encoder(smiles)
    except Exception:
        return None
    if not all(t in _VOCAB for t in sf.split_selfies(selfies_str)):
        return None
    return selfies_str


def write_jsonl(pairs_with_task: list[tuple[MolTask, dict]], output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as f_out:
        for task, pair in pairs_with_task:
            messages = {
                "chosen": make_messages(task.sequence, task.seed_selfies, pair["chosen_selfies"]),
                "rejected": make_messages(task.sequence, task.seed_selfies, pair["rejected_selfies"]),
            }
            f_out.write(json.dumps(messages) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--task-index", type=int, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--torchtune-config", type=Path, required=True, help="pi_ref's SFT config (cfg.torchtune_config)")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="pi_ref checkpoint (BOLT-<milestone>)")
    parser.add_argument("--tau-mol", type=float, default=0.4)
    parser.add_argument("--m", type=int, required=True, help="pool size -- reuses cfg.init_size")
    parser.add_argument("--target-pairs-per-task", type=int, required=True, help="stop once this many reliable pairs are found for a task")
    parser.add_argument("--max-candidates-per-task", type=int, required=True, help="give up after evaluating this many candidates, even short of target -- each candidate costs exactly --num-backgrounds real-BO calls")
    parser.add_argument("--num-backgrounds", type=int, default=8, help="M")
    parser.add_argument("--tau-q", type=float, default=1.0)
    parser.add_argument("--z-min", type=float, default=1.96)
    parser.add_argument("--delta-t", type=float, default=0.0)
    parser.add_argument("--bo-steps", type=int, choices=[0, 1], default=1, help="1 = actual one-step BO (paper's method); 0 = zero-step ablation")
    parser.add_argument(
        "--require-h0",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cumulative H>=0 rule (default): a pair is kept only if the sign(Delta_H) winner is ALSO the winner "
        "on the two candidates' own pre-rollout y. Vacuous when --bo-steps 0.",
    )
    parser.add_argument(
        "--milestone",
        type=int,
        required=True,
        help="Current BOLT-<milestone> checkpoint being used as pi_ref -- namespaces the one-step-BO work dir "
        "(mi_onestep_bo/milestone_<m>/task_<i>) so a later milestone's pair construction never reuses an earlier "
        "milestone's stale init files/trajectory CSV at the same task index.",
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--bsz", type=int, required=True, help="deployment acquisition batch size -- one-step's oracle_budget")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument(
        "--parallel-gpus", nargs="*", default=None,
        help="CUDA device ids to dispatch both reference-model scoring (mi_orpt/warm_scoring_pool.py) "
        "and one-step-BO evaluation (mi_orpt/warm_pool.py) across concurrently. Omit for serial, "
        "on this process's own device.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidate-init",
        type=Path,
        nargs="+",
        default=None,
        help="Per-task init.txt from a dedicated cfg.mi_candidate_temperature sampling pass, one per "
        "--input-csv in the same order. When given (together with --candidate-scores), the eligible "
        "bank is built from these instead of --input-csv's trajectory CSV.",
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

    # Minimal MolExperimentConfig reconstruction: construct_pairs_for_task only
    # ever reads orpt_pairs_dir/bolt_root off it (oracle_budget/init_size get
    # overridden per-pool anyway inside one_step_evaluator.py) -- milestones/
    # sft_epochs are irrelevant here since pi_ref's checkpoint is already
    # resolved (--checkpoint-dir) and no SFT/ORPT training happens in this process.
    cfg = MolExperimentConfig(
        experiment_id=args.experiment_id,
        milestones=[1],
        oracle_budget=args.bsz,
        init_size=args.m,
        bsz=args.bsz,
        sft_epochs=1,
        tau_mol=args.tau_mol,
        cuda_visible_devices=args.cuda_visible_devices,
        mi_parallel_gpus=args.parallel_gpus,
    )

    tasks = [get_task(i) for i in args.task_index]
    if len(args.input_csv) != len(tasks):
        raise ValueError(f"Expected one task per input CSV, got {len(args.input_csv)} CSVs and {len(tasks)} tasks.")

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

    # Created once, shared across every task in this milestone -- amortizes the
    # scoring model's load cost over the whole invocation, not once per task
    # (mi_orpt/warm_scoring_pool.py / likelihood.py's own docstrings).
    scoring_pool = create_scoring_pool(args.parallel_gpus, args.torchtune_config, args.checkpoint_dir) if args.parallel_gpus else None
    scoring_model = scoring_tokenizer = None
    if scoring_pool is None:
        from .likelihood import load_model_and_tokenizer, score_sequences

        scoring_model, scoring_tokenizer = load_model_and_tokenizer(args.torchtune_config, args.checkpoint_dir)

    # Separate pool, same GPU list -- one_step_evaluator.py's own module
    # docstring / this file's own module docstring explain why this stays a
    # second pool rather than reusing scoring_pool.
    worker_pool = create_mol_pool(args.parallel_gpus) if args.parallel_gpus else None

    try:
        all_pairs_with_task: list[tuple[MolTask, dict]] = []
        for input_csv, task, candidate_init, candidate_scores in zip(
            args.input_csv, tasks, args.candidate_init, args.candidate_scores
        ):
            task_context = render_context(task.sequence, task.seed_selfies)
            if candidate_init is not None:
                bank = build_eligible_bank_from_init_scores(candidate_init, candidate_scores, task.seed_smiles, args.tau_mol)
            else:
                bank = build_eligible_bank(input_csv, task.seed_smiles, args.tau_mol)

            if scoring_pool is not None:
                log_probs = score_sequences_parallel(
                    scoring_pool,
                    len(args.parallel_gpus),
                    context=task_context,
                    sequences=[c.seq for c in bank],
                    system_prompt=SYSTEM_PROMPT,
                )
            else:
                log_probs = score_sequences(
                    scoring_model,
                    scoring_tokenizer,
                    context=task_context,
                    sequences=[c.seq for c in bank],
                    system_prompt=SYSTEM_PROMPT,
                )
            log_likelihoods = {c.seq: lp for c, lp in zip(bank, log_probs)}

            work_dir_root = cfg.orpt_pairs_dir / "mi_onestep_bo" / f"milestone_{args.milestone:04d}" / f"task_{task.task_idx:04d}"
            task_pairs = construct_pairs_for_task(
                cfg=cfg,
                task_idx=task.task_idx,
                bank=bank,
                task_context=task_context,
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
                require_h0=args.require_h0,
            )
            n_selfies_reject = 0
            for p in task_pairs:
                chosen_selfies = smiles_to_selfies_vocab_checked(p["chosen_sequence"])
                rejected_selfies = smiles_to_selfies_vocab_checked(p["rejected_sequence"])
                if chosen_selfies is None or rejected_selfies is None:
                    n_selfies_reject += 1
                    continue
                p["chosen_selfies"] = chosen_selfies
                p["rejected_selfies"] = rejected_selfies
                all_pairs_with_task.append((task, p))
            if n_selfies_reject:
                print(
                    f"[build_pairs] task {task.task_idx}: {n_selfies_reject}/{len(task_pairs)} pair(s) dropped "
                    "(chosen/rejected SMILES failed SELFIES vocab re-encoding)"
                )
    finally:
        if scoring_pool is not None:
            scoring_pool.shutdown(wait=True)
        if worker_pool is not None:
            worker_pool.shutdown(wait=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=PAIRS_CSV_FIELDNAMES)
        writer.writeheader()
        for _, pair in all_pairs_with_task:
            writer.writerow({k: pair[k] for k in PAIRS_CSV_FIELDNAMES})

    write_jsonl(all_pairs_with_task, args.output_jsonl)
    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote JSONL: {args.output_jsonl}")
    print(f"Total matched-intervention DPO pairs: {len(all_pairs_with_task)}")


if __name__ == "__main__":
    main()
