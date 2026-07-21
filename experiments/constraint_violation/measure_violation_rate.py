"""Compute the similarity-constraint violation rate of raw LLM proposals,
broken down by (arm, milestone, task_set, k), from the outputs of the
existing heldout20 init_only eval and this directory's run_trainset_eval.py.

Reads task_<idx>_sampled_attempt*.jsonl files directly -- these are the
model's raw, unmodified generations. Deliberately does NOT read
task_<idx>_init.txt/_scores.csv: steps.py::_ensure_constraint_feasible
overwrites those with guaranteed-feasible mutations whenever fewer than 10%
of candidates are feasible, which would understate the real violation rate.
See /root/.claude/plans/orpt-beta-0-25-parsed-turing.md for the full
rationale.

Similarity/violation formula reused as-is from
peptide_experiment/steps.py::_ensure_constraint_feasible (edit-distance
based: similarity = (len(reference) - edit_distance(seq, reference)) /
len(reference); violation iff similarity < cfg.similarity_threshold).

Usage (run from the BOLT repo root):
    python experiments/constraint_violation/measure_violation_rate.py \\
        --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml \\
        --out-dir experiments/constraint_violation/results/peptide_100task_orpt_beta0.25
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BOLT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOLT_ROOT))
sys.path.insert(0, str(BOLT_ROOT / "fine-tuning" / "peptides" / "sampled_output_from_ft"))

from peptide_experiment.config import ExperimentConfig, load_config  # noqa: E402
from apex_oracle.refseqs import REFERENCE_SEQUENCE  # noqa: E402
from make_initialization_data import extract_sequence  # noqa: E402
from Levenshtein import distance as edit_distance  # noqa: E402

from run_trainset_eval import ARM_CHECKPOINT_FN, train_task_subset  # noqa: E402

DEFAULT_K_CHECKPOINTS = [1, 10, 50, 100, 500, 1000]


def similarity(seq: str, reference: str) -> float:
    length = len(reference)
    return (length - edit_distance(seq, reference)) / length


def load_raw_generations(task_dir: Path, task_idx: int) -> list[str]:
    attempt_files = sorted(
        task_dir.glob(f"task_{task_idx:04d}_sampled_attempt*.jsonl"),
        key=lambda p: int(p.stem.rsplit("attempt", 1)[1]),
    )
    sequences: list[str] = []
    for f in attempt_files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for answer in record.get("generated_answers", []):
                if answer is None:
                    continue
                seq = extract_sequence(str(answer))
                if seq:
                    sequences.append(seq)
    return sequences


def violation_rates_for_task(
    sequences: list[str],
    reference: str,
    similarity_threshold: float,
    k_checkpoints: list[int],
) -> dict[int, float | None]:
    rates: dict[int, float | None] = {}
    for k in k_checkpoints:
        if k > len(sequences):
            rates[k] = None
            continue
        first_k = sequences[:k]
        n_violations = sum(1 for s in first_k if similarity(s, reference) < similarity_threshold)
        rates[k] = n_violations / len(first_k)
    return rates


def task_sets(cfg: ExperimentConfig) -> dict[str, list[int]]:
    return {
        "trainset": train_task_subset(cfg),
        "heldout20": cfg.heldout_tasks("heldout20"),
    }


def task_dir_for(cfg: ExperimentConfig, task_set: str, arm: str, milestone: int) -> Path:
    if task_set == "trainset":
        return cfg.run_dir / "trainset_eval" / f"{arm}-{milestone}"
    assert task_set == "heldout20"
    return cfg.run_dir / "heldout20" / "init_only" / f"{arm}-{milestone}"


def compute_rows(
    cfg: ExperimentConfig,
    arm_prefixes: list[str],
    k_checkpoints: list[int],
    variant_label: str | None = None,
) -> list[dict]:
    """variant_label, if given, overrides the 'arm' value written to output
    rows (directory lookup / checkpoint resolution still uses the real arm
    prefix, e.g. "ORPT") -- lets a differently-trained checkpoint under the
    same arm name (e.g. a second ORPT variant trained with a different
    orpt_pairing_mode, in its own experiment_id/run_dir) be told apart once
    its CSV is concatenated with another experiment's results."""
    rows = []
    sets = task_sets(cfg)
    for milestone in cfg.milestones:
        for arm in arm_prefixes:
            for task_set, task_indices in sets.items():
                task_dir = task_dir_for(cfg, task_set, arm, milestone)
                if not task_dir.exists():
                    print(f"[measure_violation_rate] {arm}-{milestone}/{task_set}: "
                          f"no output at {task_dir}, skipping")
                    continue
                for task_idx in task_indices:
                    sequences = load_raw_generations(task_dir, task_idx)
                    if not sequences:
                        continue
                    reference = REFERENCE_SEQUENCE[task_idx]
                    rates = violation_rates_for_task(
                        sequences, reference, cfg.similarity_threshold, k_checkpoints
                    )
                    for k, rate in rates.items():
                        rows.append(
                            {
                                "arm": variant_label or arm,
                                "milestone": milestone,
                                "task_set": task_set,
                                "task_idx": task_idx,
                                "k": k,
                                "n_available": len(sequences),
                                "violation_rate": rate,
                            }
                        )
    return rows


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    for row in rows:
        if row["violation_rate"] is None:
            continue
        key = (row["arm"], row["milestone"], row["task_set"], row["k"])
        groups.setdefault(key, []).append(row["violation_rate"])
    summary = []
    for (arm, milestone, task_set, k), values in sorted(groups.items()):
        summary.append(
            {
                "arm": arm,
                "milestone": milestone,
                "task_set": task_set,
                "k": k,
                "n_tasks": len(values),
                "mean_violation_rate": sum(values) / len(values),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arms", default="BOLT,ORPT")
    parser.add_argument(
        "--k-checkpoints",
        default=",".join(str(k) for k in DEFAULT_K_CHECKPOINTS),
        help="Comma-separated proposal-count cutoffs to measure violation rate at",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write per-task.csv / summary.csv (default: "
        "experiments/constraint_violation/results/<experiment_id>)",
    )
    parser.add_argument(
        "--variant-label",
        default=None,
        help="Override the 'arm' column value in the output CSVs (checkpoint lookup still "
        "uses --arms) -- e.g. 'ORPT-LEX' for a lexicographic-pairing ORPT run in its own "
        "experiment_id, so its results stay distinguishable from another experiment's 'ORPT' "
        "rows once the two summary CSVs are concatenated for comparison.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    arm_prefixes = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arm_prefixes:
        assert a in ARM_CHECKPOINT_FN, f"unknown arm prefix {a!r}, expected one of {list(ARM_CHECKPOINT_FN)}"
    k_checkpoints = sorted(int(k.strip()) for k in args.k_checkpoints.split(",") if k.strip())

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(__file__).resolve().parent / "results" / cfg.experiment_id
    )

    rows = compute_rows(cfg, arm_prefixes, k_checkpoints, variant_label=args.variant_label)
    write_csv(
        rows,
        out_dir / "per_task_violation_rate.csv",
        fieldnames=["arm", "milestone", "task_set", "task_idx", "k", "n_available", "violation_rate"],
    )
    write_csv(
        summarize(rows),
        out_dir / "summary_violation_rate.csv",
        fieldnames=["arm", "milestone", "task_set", "k", "n_tasks", "mean_violation_rate"],
    )


if __name__ == "__main__":
    main()
