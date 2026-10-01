"""Larger-N (40 targets) version of test_oracle_model_comparison.py. The 5-6-target
runs showed high target-to-target variance in Spearman rho for both models (roughly
-0.2 to +0.6), too noisy on its own to characterize "typical" oracle quality --
this is the authoritative benchmark for the pre-scale report.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mol_oracle import MolOracle, ScoreCache  # noqa: E402

PAIRS_TSV = Path(__file__).resolve().parent / "task_manifest" / "intermediate" / "step3_pairs_with_vocab_flag.tsv"
MPNN_DIR = Path(__file__).resolve().parent / "pretrained" / "pretrained_models" / "mpnn_cnn_bindingdb_ic50"
MORGAN_DIR = Path(__file__).resolve().parent / "pretrained" / "pretrained_models" / "morgan_cnn_bindingdb_ic50"
MAX_SEQ_PROTEIN = 1000
N_TARGETS = 40
N_LIGANDS_PER_TARGET = 15


def spearman(x: list[float], y: list[float]) -> float:
    def rank(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        for rr, i in enumerate(order):
            r[i] = rr
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - (6 * d2) / (n * (n**2 - 1))


def main() -> None:
    by_target: dict[str, list[dict]] = {}
    with open(PAIRS_TSV, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["selfies_vocab_ok"] != "1":
                continue
            by_target.setdefault(row["target_id"], []).append(row)

    candidates = [(t, rows) for t, rows in by_target.items()
                  if len(rows) >= N_LIGANDS_PER_TARGET and len(rows[0]["sequence"]) <= MAX_SEQ_PROTEIN]
    rng = random.Random(7)
    rng.shuffle(candidates)
    test_targets = candidates[:N_TARGETS]
    print(f"testing {len(test_targets)} randomly-sampled targets "
          f"(from {len(candidates)} eligible with >= {N_LIGANDS_PER_TARGET} vocab-ok ligands "
          f"and sequence <= {MAX_SEQ_PROTEIN})")
    print()

    mpnn_cache = ScoreCache(Path(__file__).resolve().parent / "oracle_cache_mpnn_cnn_bindingdb_ic50.sqlite3")
    morgan_cache = ScoreCache(Path(__file__).resolve().parent / "oracle_cache_morgan_cnn_bindingdb_ic50.sqlite3")

    rhos = {"MPNN_CNN": [], "Morgan_CNN": []}
    maes = {"MPNN_CNN": [], "Morgan_CNN": []}
    for target_id, rows in test_targets:
        sample_rng = random.Random(42)
        sample = sample_rng.sample(rows, N_LIGANDS_PER_TARGET)
        seq = sample[0]["sequence"]
        measured = [float(r["p_median"]) for r in sample]
        selfies_list = [r["selfies_str"] for r in sample]

        for model_name, model_dir, cache in (
            ("MPNN_CNN", MPNN_DIR, mpnn_cache),
            ("Morgan_CNN", MORGAN_DIR, morgan_cache),
        ):
            oracle = MolOracle(target_id=target_id, target_sequence=seq, model_dir=model_dir, cache=cache)
            pred = oracle.query_oracle(selfies_list)
            rho = spearman(measured, pred)
            mae = sum(abs(m - p) for m, p in zip(measured, pred)) / len(sample)
            rhos[model_name].append(rho)
            maes[model_name].append(mae)

    print(f"{'model':<12} {'mean_rho':>9} {'median_rho':>11} {'frac_rho>0.3':>13} {'frac_rho>0.5':>13} {'mean_MAE':>9}")
    for model_name in ("MPNN_CNN", "Morgan_CNN"):
        r = sorted(rhos[model_name])
        n = len(r)
        mean_rho = sum(r) / n
        median_rho = r[n // 2]
        frac_gt_03 = sum(1 for v in r if v > 0.3) / n
        frac_gt_05 = sum(1 for v in r if v > 0.5) / n
        mean_mae = sum(maes[model_name]) / n
        print(f"{model_name:<12} {mean_rho:>9.3f} {median_rho:>11.3f} {frac_gt_03:>13.3f} {frac_gt_05:>13.3f} {mean_mae:>9.3f}")

    print()
    print("per-target rho (MPNN_CNN, Morgan_CNN):")
    for (target_id, _), m, g in zip(test_targets, rhos["MPNN_CNN"], rhos["Morgan_CNN"]):
        print(f"  {target_id}: {m:+.3f}  {g:+.3f}")


if __name__ == "__main__":
    main()
