"""tau_mol smoke study (spec: "Evaluate several plausible thresholds on training
tasks only, choosing one that gives both a usable feasible-candidate rate for
random/LLM proposals and nontrivial LOL-BO headroom. ... Lock it before held-out
evaluation").

Phase 1 (this file, cheap): for several training-task seeds, generate candidates
via mol_init_candidates and measure the feasible-fraction at each candidate tau.
Phase 2 (mol_bo_smoke.py): run an actual short constrained-BO loop at the
shortlisted tau to confirm nontrivial headroom before locking.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mol_fingerprint import tanimoto_similarity  # noqa: E402
from mol_init_candidates import generate_candidates_around_seed  # noqa: E402

MANIFEST = Path(__file__).resolve().parent / "task_manifest" / "mol_task_manifest.tsv"
TAU_CANDIDATES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
N_TASKS = 8
N_CANDIDATES_PER_TASK = 300


def main() -> None:
    with open(MANIFEST, newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["split"] == "train"]

    rng = random.Random(0)
    tasks = rng.sample(rows, N_TASKS)

    print(f"{'target_id':<10} " + " ".join(f"tau={t:.1f}" for t in TAU_CANDIDATES))
    per_tau_fracs = {t: [] for t in TAU_CANDIDATES}
    for row in tasks:
        seed_smiles = row["seed_canonical_smiles"]
        cands = generate_candidates_around_seed(
            seed_smiles, n_candidates=N_CANDIDATES_PER_TASK, rng_seed=hash(row["target_id"]) % (2**31)
        )
        sims = [tanimoto_similarity(c, seed_smiles) for c, _ in cands]
        line = f"{row['target_id']:<10} "
        for tau in TAU_CANDIDATES:
            frac = sum(1 for s in sims if s >= tau) / len(sims) if sims else 0.0
            per_tau_fracs[tau].append(frac)
            line += f"{frac:>7.3f} "
        print(line)

    print()
    print("mean feasible fraction across the 8 training tasks:")
    for tau in TAU_CANDIDATES:
        fracs = per_tau_fracs[tau]
        mean_frac = sum(fracs) / len(fracs)
        n_zero = sum(1 for f in fracs if f == 0.0)
        print(f"  tau={tau:.1f}: mean={mean_frac:.3f}  tasks_with_zero_feasible={n_zero}/{len(fracs)}")


if __name__ == "__main__":
    main()
