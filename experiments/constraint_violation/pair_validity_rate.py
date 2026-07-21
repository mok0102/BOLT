"""For a lexicographic-pairing ORPT run, what fraction of the actual DPO
training pairs are "genuinely valid" -- i.e. BOTH chosen_sequence AND
rejected_sequence satisfy the similarity constraint -- versus "mixed" pairs
where the win is purely a feasibility gate (chosen is feasible, rejected is
not), per make_dpo_train_data_csv.py::_pick_chosen_rejected()'s lexicographic
rule. Both-infeasible draws carry no signal and are never written to the
pairs CSV (see sample_pairs()), so every row is one of these two categories
-- there's a third, implicit "both infeasible" bucket only in the sense that
0 rows should ever land there; this script asserts that as a sanity check.

The pairs CSVs (runs/<experiment_id>/orpt_pairs/orpt_pairs_<milestone>.csv)
don't store a feasibility flag per side -- only reference_sequence,
chosen_sequence, rejected_sequence, chosen_score, rejected_score -- so
feasibility is recomputed here with the exact same formula used everywhere
else in this repo (make_train_data_csv.is_similar_enough, edit-distance
based) against each row's own reference_sequence.

Usage (run from the BOLT repo root):
    python experiments/constraint_violation/pair_validity_rate.py \\
        --config peptide_experiment/configs/peptide_100task_orpt_lexicographic.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

BOLT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOLT_ROOT))
sys.path.insert(0, str(BOLT_ROOT / "fine-tuning" / "peptides"))

from peptide_experiment.config import ExperimentConfig, load_config  # noqa: E402
from make_train_data_csv import is_similar_enough  # noqa: E402


def compute_rows(cfg: ExperimentConfig) -> list[dict]:
    rows = []
    for milestone in cfg.milestones:
        pairs_path = cfg.run_dir / "orpt_pairs" / f"orpt_pairs_{milestone}.csv"
        if not pairs_path.exists():
            print(f"[pair_validity_rate] milestone={milestone}: no pairs file at {pairs_path}, skipping")
            continue

        n_total = n_both_feasible = n_mixed = n_both_infeasible = 0
        with pairs_path.open(newline="") as f:
            for row in csv.DictReader(f):
                reference = row["reference_sequence"]
                chosen_feasible = is_similar_enough(row["chosen_sequence"], reference, cfg.similarity_threshold)
                rejected_feasible = is_similar_enough(row["rejected_sequence"], reference, cfg.similarity_threshold)
                n_total += 1
                if chosen_feasible and rejected_feasible:
                    n_both_feasible += 1
                elif chosen_feasible and not rejected_feasible:
                    n_mixed += 1
                else:
                    n_both_infeasible += 1

        if n_both_infeasible:
            print(
                f"[pair_validity_rate] milestone={milestone}: {n_both_infeasible} both-infeasible or "
                "rejected-feasible-only pairs found -- should be 0 under the lexicographic rule, "
                "investigate before trusting this milestone's numbers"
            )

        rows.append(
            {
                "milestone": milestone,
                "n_total": n_total,
                "n_both_feasible": n_both_feasible,
                "n_mixed": n_mixed,
                "n_unexpected": n_both_infeasible,
                "both_feasible_rate": n_both_feasible / n_total if n_total else None,
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["milestone", "n_total", "n_both_feasible", "n_mixed", "n_unexpected", "both_feasible_rate"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(__file__).resolve().parent / "results" / cfg.experiment_id
    )

    rows = compute_rows(cfg)
    write_csv(rows, out_dir / "pair_validity_rate.csv")

    if rows:
        total_pairs = sum(r["n_total"] for r in rows)
        total_both_feasible = sum(r["n_both_feasible"] for r in rows)
        print(
            f"\nOverall: {total_both_feasible}/{total_pairs} pairs ({total_both_feasible / total_pairs:.1%}) "
            "have BOTH sides feasible"
        )


if __name__ == "__main__":
    main()
