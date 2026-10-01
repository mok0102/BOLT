"""Milestone 2, step 4: target-disjoint split (leakage-safe) + seed-selection rule
exploration (training targets only) + locked-rule application + final manifest.

Order matters here and follows the spec exactly:
  1. build eligible targets (>=MIN_LIGANDS distinct SELFIES-vocab-ok ligands)
  2. group by *exact* protein sequence and split at the group level, fixed seed
     (so identical sequences under different target_ids never cross the split --
     verified in step2/step3 analysis that 33 such groups exist among the raw pool)
  3. explore the seed quantile/rank interval using TRAINING targets' potency
     distributions only, then lock one fixed rule
  4. apply the locked rule identically to every target (train AND held-out) to pick
     its seed -- never re-tuned per split, never chosen by looking at held-out data
  5. verify every chosen seed actually round-trips through the loaded VAE checkpoint
     (cheap: only ~950 forward passes, not the full candidate pool)

Never select tasks or thresholds by ORPT/BOLT performance (F7 / spec's "Task
construction" section) -- every choice below is justified from data-quality/coverage
statistics computed in this script or in step1-step3, not from downstream results.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_upstream" / "lolbo"))
import torch  # noqa: E402
import selfies as sf  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from lolbo.utils.mol_utils.selfies_vae.model_positional_unbounded import (  # noqa: E402
    SELFIESDataset, InfoTransformerVAE,
)

RDLogger.DisableLog("rdApp.*")

IN_TSV = Path(__file__).resolve().parent / "intermediate" / "step3_pairs_with_vocab_flag.tsv"
VAE_CKPT = (Path(__file__).resolve().parents[1] / "_upstream" / "lolbo" / "lolbo" / "utils"
            / "mol_utils" / "selfies_vae" / "state_dict" / "SELFIES-VAE-state-dict.pt")
OUT_DIR = Path(__file__).resolve().parents[1] / "task_manifest"
OUT_MANIFEST = OUT_DIR / "mol_task_manifest.tsv"
OUT_REPORT = OUT_DIR / "manifest_report.json"

MIN_VOCAB_OK_LIGANDS = 10          # locked in step3 analysis: 2828 eligible targets at this bar
MAX_TARGET_SEQ_LEN = 1000          # DeepPurpose.utils.trans_protein's MAX_SEQ_PROTEIN: the CNN
                                    # target encoder silently truncates anything longer than this
                                    # to its first MAX_TARGET_SEQ_LEN residues. Found while writing
                                    # test_oracle_direction.py: the oracle's within-target rank
                                    # correlation was near-zero even for random (non-adversarial)
                                    # ligand samples, on targets both above AND below this length --
                                    # so truncation does not explain that finding (see
                                    # oracle_direction_report.json for the real explanation) -- but
                                    # it is a separate, real, independent capability limit of the
                                    # frozen oracle that must be respected regardless: a target whose
                                    # sequence is truncated is one whose binding-relevant region may
                                    # not even be visible to the model, so every task must fit
                                    # entirely within what the oracle actually reads. 2436 of 2828
                                    # ligand-eligible targets satisfy this (verified before locking).
N_TRAIN_TARGET = 900               # mirrors the peptide manuscript scale
N_HELDOUT_TARGET = 50
SPLIT_RANDOM_SEED = 0
TOP_FRACTION_EXCLUDED = 0.2        # candidate seed pool excludes the top 20% most potent
SEED_RANK_QUANTILE = 0.5           # within the remaining 80%, take the one closest to its median


def main() -> None:
    # ---- load all rows, keep only vocab-ok ----
    rows_by_target: dict[str, list[dict]] = defaultdict(list)
    target_meta: dict[str, tuple[str, str]] = {}
    with open(IN_TSV, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["selfies_vocab_ok"] != "1":
                continue
            rows_by_target[row["target_id"]].append(row)
            target_meta.setdefault(row["target_id"], (row["target_name"], row["sequence"]))

    eligible_by_ligand_count = [t for t, rows in rows_by_target.items() if len(rows) >= MIN_VOCAB_OK_LIGANDS]
    eligible_targets = [t for t in eligible_by_ligand_count
                        if len(target_meta[t][1]) <= MAX_TARGET_SEQ_LEN]
    print(f"eligible by ligand count (>= {MIN_VOCAB_OK_LIGANDS} vocab-ok ligands): "
          f"{len(eligible_by_ligand_count)}")
    print(f"eligible AND sequence <= {MAX_TARGET_SEQ_LEN} residues "
          f"(oracle CNN target-encoder capability limit): {len(eligible_targets)}")

    # ---- group eligible targets by *exact* sequence (leakage guard) ----
    seq_to_targets: dict[str, list[str]] = defaultdict(list)
    for t in eligible_targets:
        _, seq = target_meta[t]
        seq_to_targets[seq].append(t)
    groups = list(seq_to_targets.values())
    n_multi_member = sum(1 for g in groups if len(g) > 1)
    print(f"sequence-groups among eligible targets: {len(groups)} "
          f"({n_multi_member} groups span >1 target_id)")

    rng = random.Random(SPLIT_RANDOM_SEED)
    rng.shuffle(groups)

    train_targets: list[str] = []
    heldout_targets: list[str] = []
    for g in groups:
        if len(train_targets) < N_TRAIN_TARGET:
            train_targets.extend(g)
        elif len(heldout_targets) < N_HELDOUT_TARGET:
            heldout_targets.extend(g)
        else:
            break  # remaining eligible targets unused; never select by downstream performance

    print(f"achieved split: {len(train_targets)} train / {len(heldout_targets)} held-out "
          f"(targets: N_TRAIN_TARGET={N_TRAIN_TARGET}, N_HELDOUT_TARGET={N_HELDOUT_TARGET}; "
          "exact counts may exceed the target by a small amount because whole "
          "sequence-groups are assigned atomically, never split)")

    # ---- explore the seed quantile rule on TRAINING targets only ----
    def candidate_pool(target_id: str) -> list[dict]:
        return sorted(rows_by_target[target_id], key=lambda r: float(r["p_median"]))
        # ascending p_median: index 0 = weakest binder, index -1 = strongest binder

    def seed_headroom_stats(target_ids: list[str], top_frac: float, rank_q: float):
        headrooms = []
        for t in target_ids:
            pool = candidate_pool(t)
            n = len(pool)
            n_excluded = max(1, int(round(n * top_frac)))
            usable = pool[: n - n_excluded]  # drop the top n_excluded strongest
            if not usable:
                continue
            idx = min(len(usable) - 1, int(round((len(usable) - 1) * rank_q)))
            seed_p = float(usable[idx]["p_median"])
            best_p = float(pool[-1]["p_median"])
            headrooms.append(best_p - seed_p)
        return headrooms

    print()
    print("=== seed-selection rule exploration on TRAINING targets only ===")
    for top_frac in (0.1, 0.2, 0.3):
        for rank_q in (0.4, 0.5, 0.6):
            hr = seed_headroom_stats(train_targets, top_frac, rank_q)
            if not hr:
                continue
            hr_sorted = sorted(hr)
            median_hr = hr_sorted[len(hr_sorted) // 2]
            frac_zero_headroom = sum(1 for h in hr if h <= 1e-9) / len(hr)
            print(f"top_frac={top_frac:.1f} rank_q={rank_q:.1f}: "
                  f"n={len(hr)} median_headroom(p)={median_hr:.3f} "
                  f"frac_zero_headroom={frac_zero_headroom:.3f}")

    # Locked rule (chosen from the exploration above, training-data-only headroom
    # numbers -- printed above for audit, not re-tuned against ORPT/BOLT results):
    print()
    print(f"LOCKED RULE: exclude top {TOP_FRACTION_EXCLUDED:.0%} strongest measured ligands; "
          f"seed = ligand at the {SEED_RANK_QUANTILE:.0%} potency rank of the remaining pool.")

    # ---- apply the locked rule to every target (train AND held-out) ----
    dataobj = SELFIESDataset()
    vae = InfoTransformerVAE(dataset=dataobj)
    state_dict = torch.load(VAE_CKPT, map_location="cpu")
    vae.load_state_dict(state_dict, strict=True)
    vae.eval()

    def pick_seed(target_id: str) -> dict:
        pool = candidate_pool(target_id)
        n = len(pool)
        n_excluded = max(1, int(round(n * TOP_FRACTION_EXCLUDED)))
        usable = pool[: n - n_excluded]
        idx = min(len(usable) - 1, int(round((len(usable) - 1) * SEED_RANK_QUANTILE)))
        seed_row = usable[idx]
        return {
            "seed_row": seed_row,
            "within_target_rank": idx + 1,          # 1-indexed, among 'usable'
            "n_usable_candidates": len(usable),
            "n_total_vocab_ok_candidates": n,
            "best_known_p_vocab_ok": float(pool[-1]["p_median"]),
        }

    def vae_round_trip_ok(selfies_str: str) -> bool:
        tokens = list(sf.split_selfies(selfies_str))
        idx_tensor = dataobj.encode(tokens).unsqueeze(0)
        with torch.no_grad():
            mu, sigma = vae.encode(idx_tensor)
            z = vae.sample_posterior(mu, sigma)
            sample_tokens = vae.sample(z=z)
        decoded_selfie = dataobj.decode(sample_tokens[0].tolist())
        try:
            decoded_smiles = sf.decoder(decoded_selfie)
        except Exception:
            return False
        return Chem.MolFromSmiles(decoded_smiles) is not None

    torch.manual_seed(SPLIT_RANDOM_SEED)

    manifest_rows = []
    n_roundtrip_fail = 0
    for split_name, target_list in (("train", train_targets), ("heldout", heldout_targets)):
        for target_id in target_list:
            target_name, sequence = target_meta[target_id]
            info = pick_seed(target_id)
            seed_row = info["seed_row"]
            rt_ok = vae_round_trip_ok(seed_row["selfies_str"])
            if not rt_ok:
                n_roundtrip_fail += 1
            manifest_rows.append({
                "split": split_name,
                "target_id": target_id,
                "target_name": target_name,
                "sequence": sequence,
                "seed_canonical_smiles": seed_row["canonical_smiles"],
                "seed_selfies": seed_row["selfies_str"],
                "seed_p_median": seed_row["p_median"],
                "seed_n_replicates": seed_row["n_replicates"],
                "within_target_rank": info["within_target_rank"],
                "n_usable_candidates": info["n_usable_candidates"],
                "n_total_vocab_ok_candidates": info["n_total_vocab_ok_candidates"],
                "best_known_p_vocab_ok": info["best_known_p_vocab_ok"],
                "seed_to_best_headroom_p": info["best_known_p_vocab_ok"] - float(seed_row["p_median"]),
                "seed_vae_roundtrip_ok": "1" if rt_ok else "0",
            })

    with open(OUT_MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    headrooms = [r["seed_to_best_headroom_p"] for r in manifest_rows]
    report = {
        "min_vocab_ok_ligands_threshold": MIN_VOCAB_OK_LIGANDS,
        "max_target_seq_len_threshold": MAX_TARGET_SEQ_LEN,
        "eligible_by_ligand_count_only": len(eligible_by_ligand_count),
        "eligible_targets_before_split": len(eligible_targets),
        "sequence_groups_among_eligible": len(groups),
        "multi_member_sequence_groups": n_multi_member,
        "split_random_seed": SPLIT_RANDOM_SEED,
        "n_train_targets_achieved": len(train_targets),
        "n_heldout_targets_achieved": len(heldout_targets),
        "locked_rule_top_fraction_excluded": TOP_FRACTION_EXCLUDED,
        "locked_rule_rank_quantile": SEED_RANK_QUANTILE,
        "seed_vae_roundtrip_failures": n_roundtrip_fail,
        "seed_to_best_headroom_p_min": min(headrooms),
        "seed_to_best_headroom_p_median": sorted(headrooms)[len(headrooms) // 2],
        "seed_to_best_headroom_p_max": max(headrooms),
    }
    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
