"""Phase 2 of the tau_mol smoke study, redesigned (Milestone 4) around the
generate-and-filter architecture: latent BO is unconstrained (pure potency; see
MOL_LATENT_SPACE_FINDING.md for why an in-loop Tanimoto constraint doesn't work in
this VAE's latent space). Feasibility is enforced by (1) building the init pool
only from Tanimoto-pre-filtered candidates and (2) masking BO's own trajectory by
Tanimoto-to-seed when computing the reported "best feasible score" -- mirroring
peptide's experiments/eval2/domains/peptide.py::best_objective_at_k exactly (never
report an unmasked max as the best result).

Minimal driver -- LOLBOState exposes .acquisition()/.update_surrogate_model() as
per-step primitives with no built-in run loop; this mirrors
optimization/peptides/lolbo_scripts/optimize.py::run_lolbo's real per-step call
sequence (read-only reference, not imported) -- omitting update_surrogate_model()
was tried first and silently froze the GP at its initial fit for the whole run.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mol_lolbo.lolbo import LOLBOState  # noqa: E402
from mol_init_candidates import generate_feasible_candidates_around_seed  # noqa: E402
from mol_objective import MoleculeObjective  # noqa: E402
from mol_oracle import EnsembleMolOracle  # noqa: E402
from mol_fingerprint import tanimoto_similarity  # noqa: E402
from mol_acquisition import mol_acquisition_step  # noqa: E402

MANIFEST = Path(__file__).resolve().parent / "task_manifest" / "mol_task_manifest.tsv"


def load_task(target_id: str) -> dict:
    with open(MANIFEST, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["target_id"] == target_id:
                return row
    raise KeyError(target_id)


def best_feasible_score(train_x: list[str], train_y: torch.Tensor, seed_smiles: str, tau_mol: float) -> float | None:
    """Post-hoc masked reduction: the analogue of
    experiments/eval2/domains/peptide.py::best_objective_at_k. Never an unmasked
    max -- an out-of-neighborhood candidate the unconstrained oracle happens to
    score well is exactly the failure mode that function's docstring documents."""
    best = None
    for x, y in zip(train_x, train_y.squeeze(-1).tolist()):
        sim = tanimoto_similarity(x, seed_smiles)
        if sim is not None and sim >= tau_mol:
            if best is None or y > best:
                best = y
    return best


def run_smoke(target_id: str, tau_mol: float, n_init: int, n_bo_steps: int, bsz: int) -> dict:
    row = load_task(target_id)
    seed_smiles = row["seed_canonical_smiles"]

    oracle = EnsembleMolOracle(target_id=target_id, target_sequence=row["sequence"])
    objective = MoleculeObjective(oracle=oracle, seed_smiles=seed_smiles, task_id=target_id)  # unconstrained

    init_cands, gen_stats = generate_feasible_candidates_around_seed(
        seed_smiles, tau_mol=tau_mol, n_feasible=n_init - 1, rng_seed=0
    )
    print(f"  [{target_id} tau={tau_mol}] init candidate generation: {gen_stats}")
    init_smiles = [seed_smiles] + [c for c, _ in init_cands]

    t0 = time.time()
    init_scores = objective.query_oracle(init_smiles)
    print(f"  [{target_id} tau={tau_mol}] scored {len(init_smiles)} init candidates in {time.time()-t0:.1f}s")

    keep = [i for i, s in enumerate(init_scores) if s == s]  # drop NaN
    init_smiles = [init_smiles[i] for i in keep]
    init_scores = [init_scores[i] for i in keep]
    objective.xs_to_scores_dict = dict(zip(init_smiles, init_scores))

    init_z, _ = objective.vae_forward(init_smiles)
    init_z = init_z.detach().cpu()
    init_y = torch.tensor(init_scores, dtype=torch.float32).unsqueeze(-1)

    lolbo_state = LOLBOState(
        objective=objective,
        surrogate_type="gp_dkl",
        train_x=init_smiles,
        train_y=init_y,
        train_z=init_z,
        train_c=None,  # unconstrained -- see module docstring
        minimize=False,
        bsz=bsz,
        k=10,
        verbose=False,
    )

    best_feasible_at_init = best_feasible_score(init_smiles, init_y, seed_smiles, tau_mol)
    n_feasible_init = sum(1 for x in init_smiles if (s := tanimoto_similarity(x, seed_smiles)) is not None and s >= tau_mol)
    print(f"  [{target_id} tau={tau_mol}] init: {n_feasible_init}/{len(init_smiles)} feasible "
          f"(should be ~{len(init_smiles)}, generation already pre-filtered), "
          f"best feasible score = {best_feasible_at_init}")

    # mol_acquisition_step replaces LOLBOState.acquisition() (see its module
    # docstring / MOL_LATENT_SPACE_FINDING.md for why): candidates are sourced from
    # fresh Tanimoto-feasible SELFIES edits, not Sobol-perturb-then-decode, so the
    # GP only ever has to learn z -> potency, never z -> feasibility. update_next()
    # itself (bookkeeping: incumbent tracking, train_x/y/z growth, TuRBO
    # success/failure) is untouched, byte-identical ported code.
    t0 = time.time()
    gen_yields = []
    for step in range(n_bo_steps):
        lolbo_state.update_surrogate_model()
        gen_stats_round = mol_acquisition_step(
            lolbo_state, seed_smiles=seed_smiles, tau_mol=tau_mol,
            n_candidates=max(20, bsz * 4), rng_seed=1000 + step,
        )
        gen_yields.append(gen_stats_round.get("yield_feasible_per_attempt"))
        if lolbo_state.tr_state.restart_triggered:
            lolbo_state.initialize_tr_state()
    mean_yield = sum(y for y in gen_yields if y is not None) / max(1, sum(1 for y in gen_yields if y is not None))
    print(f"  [{target_id} tau={tau_mol}] {n_bo_steps} acquisition steps in {time.time()-t0:.1f}s, "
          f"mean per-round generation yield={mean_yield:.3f}")

    best_feasible_after = best_feasible_score(lolbo_state.train_x, lolbo_state.train_y, seed_smiles, tau_mol)
    n_feasible_total = sum(1 for x in lolbo_state.train_x
                           if (s := tanimoto_similarity(x, seed_smiles)) is not None and s >= tau_mol)
    n_total = len(lolbo_state.train_x)

    return {
        "target_id": target_id,
        "tau_mol": tau_mol,
        "best_feasible_score_at_init": best_feasible_at_init,
        "best_feasible_score_after_bo": best_feasible_after,
        "headroom": (best_feasible_after - best_feasible_at_init)
                    if (best_feasible_after is not None and best_feasible_at_init is not None) else None,
        "n_feasible_total": n_feasible_total,
        "n_total_evaluated": n_total,
        "init_generation_stats": gen_stats,
    }


if __name__ == "__main__":
    import json

    for tau in [0.4]:
        for target_id in ["Q8WTS6"]:
            r = run_smoke(target_id, tau, n_init=100, n_bo_steps=100, bsz=5)
            print(json.dumps(r, indent=2))
