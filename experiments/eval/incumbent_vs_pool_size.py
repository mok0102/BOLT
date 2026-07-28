"""(2.1) Proposal-level performance vs. pool size, no BO: for each arm/
milestone in a manifest, at each n_proposals (number of accepted/feasible
proposals), report (a) the incumbent -- best (lowest) MIC among the first
n_proposals feasible proposals -- and (b) the rejection ratio needed to draw
that many, i.e. how many raw (possibly infeasible/duplicate) generations it
took to accumulate n_proposals feasible unique candidates.

Separate rows per task_set (trainset / heldout) so trained-peptide and
held-out-peptide performance can be reported/plotted independently.

Reads raw task_<idx>_sampled_attempt*.jsonl via
common.raw_dir_for(spec, task_set) -- run generate_raw_proposals.py first
for any (arm, milestone, task_set) not yet covered. Scores are computed
fresh here via apex_wrapper (raw jsonls carry no scores, and any
task_XXXX_scores.csv on disk would be the constraint-patched/padded version,
wrong for "first n_proposals feasible raw proposals").

Usage (run from the BOLT repo root):
    python experiments/eval/incumbent_vs_pool_size.py \\
        --config peptide_experiment/configs/peptide_poc20_bolt.yaml \\
        --manifest experiments/eval/manifests/poc20_four_arm.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import ModelSpec, feasible_pool_with_draw_counts, load_manifest, raw_dir_for, task_indices, write_csv

from apex_oracle import apex_wrapper
from peptide_experiment.config import ExperimentConfig, load_config

DEFAULT_N_PROPOSALS_CHECKPOINTS = [1, 5, 10, 20, 50]

PER_TASK_FIELDS = [
    "arm", "milestone", "task_set", "task_idx", "n_proposals",
    "n_raw_total", "n_feasible_total", "draws_to_n_proposals", "rejection_rate_at_n_proposals", "incumbent_mic",
]
SUMMARY_FIELDS = [
    "arm", "milestone", "task_set", "n_proposals",
    "n_tasks", "coverage_rate_at_n_proposals", "mean_incumbent_mic", "mean_rejection_rate_at_n_proposals",
]


def compute_rows_for_task(
    cfg: ExperimentConfig, spec: ModelSpec, task_set: str, task_idx: int,
    n_proposals_checkpoints: list[int], raw_dir: Path,
) -> list[dict]:
    n_raw_total = len(list(raw_dir.glob(f"task_{task_idx:04d}_sampled_attempt*.jsonl")))
    pool = feasible_pool_with_draw_counts(cfg, task_idx, raw_dir, max_needed=max(n_proposals_checkpoints))
    seqs = [s for s, _ in pool]
    scores = list(-apex_wrapper(seqs)[:, 0]) if seqs else []

    rows = []
    for n_proposals in n_proposals_checkpoints:
        if len(pool) < n_proposals:
            rows.append({
                "arm": spec.arm, "milestone": spec.milestone, "task_set": task_set, "task_idx": task_idx,
                "n_proposals": n_proposals, "n_raw_total": n_raw_total, "n_feasible_total": len(pool),
                "draws_to_n_proposals": None, "rejection_rate_at_n_proposals": None, "incumbent_mic": None,
            })
            continue
        draws_to_n_proposals = pool[n_proposals - 1][1]
        rows.append({
            "arm": spec.arm, "milestone": spec.milestone, "task_set": task_set, "task_idx": task_idx,
            "n_proposals": n_proposals, "n_raw_total": n_raw_total, "n_feasible_total": len(pool),
            "draws_to_n_proposals": draws_to_n_proposals,
            "rejection_rate_at_n_proposals": 1 - n_proposals / draws_to_n_proposals,
            "incumbent_mic": -max(scores[:n_proposals]),
        })
    return rows


def compute_rows(
    cfg: ExperimentConfig, specs: list[ModelSpec], task_set_names: list[str], n_proposals_checkpoints: list[int],
) -> list[dict]:
    rows = []
    for spec in specs:
        for task_set in task_set_names:
            raw_dir = raw_dir_for(spec, task_set)
            if not raw_dir.exists():
                print(f"[incumbent_vs_pool_size] {spec.arm}-{spec.milestone}/{task_set}: "
                      f"no raw generations at {raw_dir}, run generate_raw_proposals.py first, skipping")
                continue
            for task_idx in task_indices(cfg, task_set):
                rows.extend(compute_rows_for_task(cfg, spec, task_set, task_idx, n_proposals_checkpoints, raw_dir))
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["arm"], row["milestone"], row["task_set"], row["n_proposals"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (arm, milestone, task_set, n_proposals), group_rows in sorted(groups.items()):
        sufficient = [r for r in group_rows if r["incumbent_mic"] is not None]
        summary.append({
            "arm": arm, "milestone": milestone, "task_set": task_set, "n_proposals": n_proposals,
            "n_tasks": len(group_rows),
            "coverage_rate_at_n_proposals": len(sufficient) / len(group_rows) if group_rows else None,
            "mean_incumbent_mic": (
                sum(r["incumbent_mic"] for r in sufficient) / len(sufficient) if sufficient else None
            ),
            "mean_rejection_rate_at_n_proposals": (
                sum(r["rejection_rate_at_n_proposals"] for r in sufficient) / len(sufficient) if sufficient else None
            ),
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-sets", default="trainset,heldout")
    parser.add_argument(
        "--n-proposals-checkpoints",
        default=",".join(str(n) for n in DEFAULT_N_PROPOSALS_CHECKPOINTS),
        help="Comma-separated accepted-proposal-count cutoffs",
    )
    parser.add_argument("--out-dir", default=None, help="Default: experiments/eval/results/<manifest stem>")
    args = parser.parse_args()

    cfg = load_config(args.config)
    specs = load_manifest(args.manifest)
    task_set_names = [t.strip() for t in args.task_sets.split(",") if t.strip()]
    n_proposals_checkpoints = sorted(
        int(n.strip()) for n in args.n_proposals_checkpoints.split(",") if n.strip()
    )

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else Path(__file__).resolve().parent / "results" / Path(args.manifest).stem
    )

    rows = compute_rows(cfg, specs, task_set_names, n_proposals_checkpoints)
    write_csv(rows, out_dir / "per_task_incumbent_vs_pool_size.csv", fieldnames=PER_TASK_FIELDS)
    write_csv(summarize(rows), out_dir / "summary_incumbent_vs_pool_size.csv", fieldnames=SUMMARY_FIELDS)


if __name__ == "__main__":
    main()
