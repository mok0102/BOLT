"""Head-to-head comparison of MPNN_CNN_BindingDB_IC50 vs Morgan_CNN_BindingDB_IC50
on the exact same sampled (target, ligand) pairs, both restricted to targets within
the oracle's MAX_SEQ_PROTEIN=1000 capability limit.

Triggered by test_oracle_direction.py finding near-zero/negative within-target rank
correlation for MPNN_CNN across both extreme-percentile and random sampling, on
targets both above and below the truncation length -- ruling out truncation as the
cause. This checks whether that is an MPNN-specific weakness or a general property
of BindingDB-pretrained DeepPurpose models.
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
                  if len(rows) >= 15 and len(rows[0]["sequence"]) <= MAX_SEQ_PROTEIN]
    rng = random.Random(42)  # same seed test_oracle_direction.py's random-sampling variant used
    rng.shuffle(candidates)
    test_targets = candidates[:6]

    mpnn_cache = ScoreCache(Path(__file__).resolve().parent / "oracle_cache_mpnn_cnn_bindingdb_ic50.sqlite3")
    morgan_cache = ScoreCache(Path(__file__).resolve().parent / "oracle_cache_morgan_cnn_bindingdb_ic50.sqlite3")

    results = {"MPNN_CNN": [], "Morgan_CNN": []}
    for target_id, rows in test_targets:
        sample_rng = random.Random(42)  # identical draw for both models
        sample = sample_rng.sample(rows, min(15, len(rows)))
        seq = sample[0]["sequence"]
        name = sample[0]["target_name"]
        measured = [float(r["p_median"]) for r in sample]
        selfies_list = [r["selfies_str"] for r in sample]

        mpnn_oracle = MolOracle(target_id=target_id, target_sequence=seq, model_dir=MPNN_DIR, cache=mpnn_cache)
        morgan_oracle = MolOracle(target_id=target_id, target_sequence=seq, model_dir=MORGAN_DIR, cache=morgan_cache)

        mpnn_pred = mpnn_oracle.query_oracle(selfies_list)
        morgan_pred = morgan_oracle.query_oracle(selfies_list)

        mpnn_rho = spearman(measured, mpnn_pred)
        morgan_rho = spearman(measured, morgan_pred)
        mpnn_mae = sum(abs(m - p) for m, p in zip(measured, mpnn_pred)) / len(sample)
        morgan_mae = sum(abs(m - p) for m, p in zip(measured, morgan_pred)) / len(sample)

        results["MPNN_CNN"].append(mpnn_rho)
        results["Morgan_CNN"].append(morgan_rho)

        print(f"{target_id} ({name!r}, seqlen={len(seq)}, n={len(sample)}):")
        print(f"    MPNN_CNN:   rho={mpnn_rho:+.3f}  MAE={mpnn_mae:.3f}")
        print(f"    Morgan_CNN: rho={morgan_rho:+.3f}  MAE={morgan_mae:.3f}")

    print()
    for model, rhos in results.items():
        mean_rho = sum(rhos) / len(rhos)
        print(f"{model}: mean rho across {len(rhos)} targets = {mean_rho:+.3f}  (individual: {[round(r,3) for r in rhos]})")


if __name__ == "__main__":
    main()
