"""Mol-specific acquisition step, replacing LOLBOState.acquisition() for this
domain. Reuses LOLBOState.update_next() (byte-identical ported bookkeeping --
incumbent tracking, train_x/y/z growth, TuRBO success/failure updates) unchanged;
only candidate *sourcing* is domain-specific.

Why this exists (see MOL_LATENT_SPACE_FINDING.md): vanilla generate_batch()
(mol_lolbo/utils/bo_utils/turbo.py, an untouched byte-identical port) sources its
Thompson-sampling candidate pool by Sobol-perturbing around the trust-region
center in raw z-space, then decodes each perturbed z through the VAE. That
candidate-sourcing step is exactly what this VAE's latent space can't support for
this domain: perturbed-then-decoded candidates land, on average, no closer to the
seed than two unrelated molecules (Tanimoto ~0.15, background ~0.14), and a
constrained-BO smoke run confirmed zero feasible candidates in 600 real
evaluations. Swapping the constraint out for post-hoc masking alone doesn't fix
this either -- unconstrained latent BO's own proposals are then almost never in
the lead-like neighborhood at all, so BO contributes no improvement over the
init pool (also confirmed empirically), and unconstrained wandering re-introduces
exactly the "unconstrained de-novo generation / OOD oracle exploitation" failure
mode the domain's lead-optimization framing exists to avoid.

The fix keeps latent BO doing real, adaptive work (GP models z -> potency, which
*is* plausible/learnable -- that's what the original LOL-BO benchmarks validated)
by changing only where its candidate pool comes from: instead of
Sobol-perturb-then-decode, generate fresh Tanimoto-feasible SELFIES-edit
candidates directly (the one generator that has actually been shown to preserve
seed-locality), encode them, and Thompson-sample among their *known* (x, z) pairs.
Because x is known by construction, there is no decode step here at all -- the
VAE's own decode noise (which is what breaks locality) never enters the loop.
This is not a new BO engine: Thompson-sampling over a finite candidate pool is
exactly what vanilla generate_batch() already does (X_cand there is finite too,
just Sobol-sourced); only the candidate source changed.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mol_lolbo.utils.bo_utils.constrained_max_posterior_sampling import MaxPosteriorSampling  # noqa: E402
from mol_init_candidates import filter_vocab_safe, generate_feasible_candidates_around_seed  # noqa: E402


def mol_acquisition_step(
    lolbo_state,
    seed_smiles: str,
    tau_mol: float,
    n_candidates: int,
    rng_seed: int,
    generation_pool: ProcessPoolExecutor | None = None,
    n_generation_workers: int = 1,
) -> dict:
    """One round: generate n_candidates fresh feasible SELFIES-edit candidates,
    encode them, Thompson-sample lolbo_state.bsz of them by the current GP
    surrogate's posterior, score the selected ones with the real oracle, and feed
    the result through the existing (unmodified) update_next() bookkeeping.
    Returns generation stats (yield -- a low yield here is reportable, per spec,
    not something to mask by silently retrying more).

    generation_pool: an additive, opt-in knob (default None -- current serial
    behavior). When given (a persistent CPU pool created once per task by the
    caller, mol_generation_pool.create_generation_pool), splits the generation
    call across n_generation_workers instead of running it serially -- see
    mol_generation_pool.py's own docstring for the profiling that motivated
    this."""
    objective = lolbo_state.objective

    if generation_pool is not None:
        from mol_generation_pool import generate_feasible_candidates_parallel

        cands, gen_stats = generate_feasible_candidates_parallel(
            generation_pool, seed_smiles, tau_mol, n_candidates, rng_seed, n_generation_workers
        )
    else:
        cands, gen_stats = generate_feasible_candidates_around_seed(
            seed_smiles, tau_mol=tau_mol, n_feasible=n_candidates, rng_seed=rng_seed
        )
    # Dedup against x's already in this task's dataset -- no point spending GP/
    # oracle effort re-considering something already evaluated this run.
    already_seen = set(lolbo_state.train_x)
    cand_smiles = [c for c, _ in cands if c not in already_seen]
    if not cand_smiles:
        gen_stats["note"] = "all generated candidates already evaluated this run"
        return gen_stats

    # Defense-in-depth final gate before vae_forward -- see
    # mol_init_candidates.filter_vocab_safe's own docstring for the incident
    # this guards against. Redundant with generate_feasible_candidates_around_seed/
    # _parallel's own (already-correct) vocab-check, kept anyway so every
    # vae_forward call site has the same safety net regardless of which
    # candidate source feeds it.
    vocab_mask = filter_vocab_safe(cand_smiles)
    if not all(vocab_mask):
        n_dropped = len(vocab_mask) - sum(vocab_mask)
        gen_stats["n_dropped_vocab_unsafe"] = n_dropped
        cand_smiles = [s for s, keep in zip(cand_smiles, vocab_mask) if keep]
    if not cand_smiles:
        gen_stats["note"] = "all generated candidates failed the vocab safety check"
        return gen_stats

    z_cand, _ = objective.vae_forward(cand_smiles)
    z_cand = z_cand.detach()

    thompson_sampling = MaxPosteriorSampling(model=lolbo_state.model, replacement=False, constrained=False)
    with torch.no_grad():
        z_selected = thompson_sampling(z_cand.cuda(), num_samples=min(lolbo_state.bsz, len(cand_smiles)))

    # Map each selected z row back to its known source SMILES by exact match
    # (z_cand rows are unique per distinct input SMILES barring VAE posterior-
    # sampling collisions, which would be a separate, reportable anomaly, not
    # silently resolved here).
    selected_smiles = []
    z_cand_cpu = z_cand.cpu()
    for row in z_selected.cpu():
        matches = (z_cand_cpu == row).all(dim=-1).nonzero(as_tuple=True)[0]
        if len(matches) == 0:
            raise RuntimeError("Thompson-sampled z row does not match any candidate's encoding -- "
                               "MaxPosteriorSampling must select from the exact rows given it.")
        selected_smiles.append(cand_smiles[matches[0].item()])

    y_selected = objective.query_oracle(selected_smiles)
    keep = [i for i, y in enumerate(y_selected) if y == y]  # drop NaN (F1/F5: never scored as 0)
    if not keep:
        gen_stats["note"] = "every Thompson-sampled candidate scored NaN"
        return gen_stats

    z_final = z_selected[keep]
    y_final = torch.tensor([y_selected[i] for i in keep], dtype=torch.float32)
    x_final = [selected_smiles[i] for i in keep]
    for smi, y in zip(x_final, y_final.tolist()):
        objective.xs_to_scores_dict[smi] = y
        objective.num_calls += 1  # every candidate here is freshly computed and non-NaN by construction (kept above)

    lolbo_state.update_next(z_final, y_final, x_final, c_next_=None, acquisition=True)
    return gen_stats
