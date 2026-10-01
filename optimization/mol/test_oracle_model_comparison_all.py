"""All 5 pretrained BindingDB-IC50 DeepPurpose models (the full available roster,
verified in Milestone 1/3: only these 5 carry the _ic50 label specifically, so they
share the same p = -log10(IC50_M) training target and are a fair apples-to-apples
comparison) plus simple unweighted-mean ensembles, on the same 40 random targets x
15 ligands used in test_oracle_model_comparison_large.py (MPNN_CNN mean rho=0.036,
Morgan_CNN mean rho=0.196 there).

Ensemble mirrors peptide's APEX oracle: an unweighted mean across independently
trained models (not weighted by this same validation run's own results, which would
be circular/overfit to this test).
"""

from __future__ import annotations

import csv
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mol_oracle import MolOracle, ScoreCache  # noqa: E402

PAIRS_TSV = Path(__file__).resolve().parent / "task_manifest" / "intermediate" / "step3_pairs_with_vocab_flag.tsv"
PRETRAINED_DIR = Path(__file__).resolve().parent / "pretrained" / "pretrained_models"
MODELS = ["mpnn_cnn_bindingdb_ic50", "morgan_cnn_bindingdb_ic50", "cnn_cnn_bindingdb_ic50",
          "daylight_aac_bindingdb_ic50", "morgan_aac_bindingdb_ic50"]
MAX_SEQ_PROTEIN = 1000
N_TARGETS = 40
N_LIGANDS_PER_TARGET = 15
RNG_TARGET_SEED = 7       # same as test_oracle_model_comparison_large.py, for an identical target set
RNG_LIGAND_SEED = 42


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
    rng = random.Random(RNG_TARGET_SEED)
    rng.shuffle(candidates)
    test_targets = candidates[:N_TARGETS]

    caches = {m: ScoreCache(Path(__file__).resolve().parent / f"oracle_cache_{m}.sqlite3") for m in MODELS}

    # per_target_scores[model][target_id] = list of predicted p aligned with `measured`
    per_target_measured: dict[str, list[float]] = {}
    per_target_pred: dict[str, dict[str, list[float]]] = {m: {} for m in MODELS}

    for target_id, rows in test_targets:
        sample_rng = random.Random(RNG_LIGAND_SEED)
        sample = sample_rng.sample(rows, N_LIGANDS_PER_TARGET)
        seq = sample[0]["sequence"]
        measured = [float(r["p_median"]) for r in sample]
        selfies_list = [r["selfies_str"] for r in sample]
        per_target_measured[target_id] = measured

        for model in MODELS:
            oracle = MolOracle(target_id=target_id, target_sequence=seq,
                                model_dir=PRETRAINED_DIR / model, cache=caches[model])
            per_target_pred[model][target_id] = oracle.query_oracle(selfies_list)

    def summarize(name: str, pred_by_target: dict[str, list[float]]) -> dict:
        rhos = [spearman(per_target_measured[t], pred_by_target[t]) for t in pred_by_target]
        maes = [sum(abs(m - p) for m, p in zip(per_target_measured[t], pred_by_target[t])) / len(pred_by_target[t])
                for t in pred_by_target]
        r = sorted(rhos)
        n = len(r)
        return {
            "name": name,
            "mean_rho": sum(r) / n,
            "median_rho": r[n // 2],
            "frac_rho_gt_0.3": sum(1 for v in r if v > 0.3) / n,
            "frac_rho_gt_0.5": sum(1 for v in r if v > 0.5) / n,
            "mean_mae": sum(maes) / n,
        }

    results = [summarize(m, per_target_pred[m]) for m in MODELS]

    # Ensembles: unweighted mean of per-candidate predicted p across a set of models,
    # per target (nan-safe: a model's nan for a given candidate excludes only that
    # model from that candidate's average, mirroring how a real BO run would combine
    # scores from several frozen oracles).
    def ensemble_pred(models: list[str]) -> dict[str, list[float]]:
        out = {}
        for t in per_target_measured:
            n_cand = len(per_target_measured[t])
            avg = []
            for i in range(n_cand):
                vals = [per_target_pred[m][t][i] for m in models if per_target_pred[m][t][i] == per_target_pred[m][t][i]]
                avg.append(sum(vals) / len(vals) if vals else float("nan"))
            out[t] = avg
        return out

    ensembles = {
        "ensemble_ALL_5": MODELS,
        "ensemble_top2_MPNN_Morgan": ["mpnn_cnn_bindingdb_ic50", "morgan_cnn_bindingdb_ic50"],
        "ensemble_no_MPNN": [m for m in MODELS if m != "mpnn_cnn_bindingdb_ic50"],
    }
    for ens_name, members in ensembles.items():
        results.append(summarize(ens_name, ensemble_pred(members)))

    print(f"{'model':<28} {'mean_rho':>9} {'median_rho':>11} {'rho>0.3':>8} {'rho>0.5':>8} {'mean_MAE':>9}")
    for r in results:
        print(f"{r['name']:<28} {r['mean_rho']:>9.3f} {r['median_rho']:>11.3f} "
              f"{r['frac_rho_gt_0.3']:>8.3f} {r['frac_rho_gt_0.5']:>8.3f} {r['mean_mae']:>9.3f}")


if __name__ == "__main__":
    main()
