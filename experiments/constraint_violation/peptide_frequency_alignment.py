"""Rejection-sample from a single checkpoint's raw generations for one task
(one target peptide) and check whether the *sampling frequency* of a
surviving candidate aligns with its APEX score -- does BOLT-<m>/ORPT-<m>
resample the same good-scoring mutation repeatedly, or is a high sampling
frequency uncorrelated with score?

This is a different question from measure_violation_rate.py/rank_alignment.py:
those measure per-task violation rate and score-vs-rank among the deduped
pool. Here we deliberately do NOT dedup first -- load_raw_generations()
already returns the raw, duplicated draws (e.g. 2 attempt files x 500
answers = ~1000 draws for one task), so counting occurrences of each exact
sequence string gives the empirical resampling frequency. Rejection
sampling = keep only draws whose sequence passes the same similarity
constraint as everywhere else in this repo (steps.py::_ensure_constraint_feasible);
infeasible draws are discarded outright, not counted at all (real
reject-and-discard, no synthetic top-up, matching run_rejection_sampled_bo.py's
approach) -- the whole point is to see the frequency *among survivors*.

Reuses already-generated raw output (task_dir_for / load_raw_generations from
measure_violation_rate.py) -- run run_trainset_eval.py / heldout_eval's
init_only eval first for whichever (arm, milestone, task_set) you want.

Usage (run from the BOLT repo root):
    python experiments/constraint_violation/peptide_frequency_alignment.py \\
        --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml \\
        --milestone 100 --task-set heldout20
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

BOLT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOLT_ROOT))
sys.path.insert(0, str(BOLT_ROOT / "fine-tuning" / "peptides" / "sampled_output_from_ft"))

from peptide_experiment.config import ExperimentConfig, load_config  # noqa: E402
from apex_oracle import apex_wrapper  # noqa: E402
from apex_oracle.refseqs import REFERENCE_SEQUENCE  # noqa: E402

from run_trainset_eval import ARM_CHECKPOINT_FN  # noqa: E402
from measure_violation_rate import load_raw_generations, similarity, task_dir_for, task_sets  # noqa: E402


def rejection_sample_frequencies(
    cfg: ExperimentConfig, task_idx: int, task_dir: Path
) -> tuple[list[tuple[str, int, float]], int]:
    """Returns ([(sequence, frequency, score), ...] sorted by frequency desc,
    n_total_draws) for one task's raw (non-deduped) generations, keeping only
    similarity-feasible sequences (rejection sampling)."""
    draws = load_raw_generations(task_dir, task_idx)
    if not draws:
        return [], 0

    reference = REFERENCE_SEQUENCE[task_idx]
    counts = Counter(draws)
    feasible_seqs = [seq for seq in counts if similarity(seq, reference) >= cfg.similarity_threshold]
    if not feasible_seqs:
        return [], len(draws)

    scores = -apex_wrapper(feasible_seqs)[:, 0]  # same sign convention as steps.py (maximize)
    ranked = sorted(
        zip(feasible_seqs, scores),
        key=lambda pair: (counts[pair[0]], pair[1]),
        reverse=True,
    )
    return [(seq, counts[seq], float(score)) for seq, score in ranked], len(draws)


def compute_rows(
    cfg: ExperimentConfig,
    arm_prefixes: list[str],
    task_set: str,
    milestone: int,
    task_indices: list[int] | None,
) -> tuple[list[dict], list[dict]]:
    """Returns (per_sequence_rows, per_task_summary_rows)."""
    sequence_rows, summary_rows = [], []
    indices = task_indices if task_indices is not None else task_sets(cfg)[task_set]

    for arm in arm_prefixes:
        task_dir = task_dir_for(cfg, task_set, arm, milestone)
        if not task_dir.exists():
            print(f"[peptide_frequency_alignment] {arm}-{milestone}/{task_set}: no output at {task_dir}, skipping")
            continue
        for task_idx in indices:
            ranked, n_total_draws = rejection_sample_frequencies(cfg, task_idx, task_dir)
            if n_total_draws == 0:
                continue
            n_accepted_draws = sum(freq for _, freq, _ in ranked)
            summary_rows.append(
                {
                    "arm": arm,
                    "milestone": milestone,
                    "task_set": task_set,
                    "task_idx": task_idx,
                    "n_total_draws": n_total_draws,
                    "n_accepted_draws": n_accepted_draws,
                    "n_unique_feasible": len(ranked),
                    "rejection_rate": (n_total_draws - n_accepted_draws) / n_total_draws,
                }
            )
            for freq_rank, (seq, freq, score) in enumerate(ranked, start=1):
                sequence_rows.append(
                    {
                        "arm": arm,
                        "milestone": milestone,
                        "task_set": task_set,
                        "task_idx": task_idx,
                        "freq_rank": freq_rank,
                        "sequence": seq,
                        "frequency": freq,
                        "frequency_fraction": freq / n_total_draws,
                        "score": score,
                    }
                )
    return sequence_rows, summary_rows


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arms", default="BOLT,ORPT")
    parser.add_argument("--task-set", default="heldout20", choices=("trainset", "heldout20"))
    parser.add_argument(
        "--milestone",
        type=int,
        default=None,
        help="Which milestone's checkpoint to use (default: the largest in cfg.milestones, "
        "i.e. the fully-trained model, e.g. BOLT-100/ORPT-100)",
    )
    parser.add_argument(
        "--task-idx",
        default=None,
        help="Comma-separated subset of task indices to process (default: every task in "
        "--task-set). Use this to keep the run cheap while sanity-checking one example task.",
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    arm_prefixes = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arm_prefixes:
        assert a in ARM_CHECKPOINT_FN, f"unknown arm prefix {a!r}, expected one of {list(ARM_CHECKPOINT_FN)}"
    milestone = args.milestone if args.milestone is not None else max(cfg.milestones)
    task_indices = (
        [int(t.strip()) for t in args.task_idx.split(",") if t.strip()] if args.task_idx else None
    )

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(__file__).resolve().parent / "results" / cfg.experiment_id
    )

    sequence_rows, summary_rows = compute_rows(cfg, arm_prefixes, args.task_set, milestone, task_indices)
    write_csv(
        sequence_rows,
        out_dir / "per_sequence_frequency_alignment.csv",
        fieldnames=["arm", "milestone", "task_set", "task_idx", "freq_rank", "sequence", "frequency",
                    "frequency_fraction", "score"],
    )
    write_csv(
        summary_rows,
        out_dir / "per_task_frequency_alignment_summary.csv",
        fieldnames=["arm", "milestone", "task_set", "task_idx", "n_total_draws", "n_accepted_draws",
                    "n_unique_feasible", "rejection_rate"],
    )


if __name__ == "__main__":
    main()
