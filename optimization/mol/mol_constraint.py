"""Molecular feasibility constraint: Tanimoto(candidate, seed) >= tau_mol, wired as
a genuine BO constraint (c(x) <= 0 form fed to a GP constraint surrogate), never a
proposal-time rejection filter. Mirrors
optimization/peptides/your_tasks/your_blackbox_constraints.py::ConstraintFunction
/ APEXSimilarityConstraint (read-only reference; that file is under
optimization/peptides/ and is never imported from here) -- ported here because the
c(x) <= 0 wrapper convention itself is domain-generic, only the underlying
similarity metric (Tanimoto vs. Levenshtein) differs.

tau_mol is NOT peptide's 0.75: fingerprint Tanimoto and normalized edit distance
are not on the same scale. See mol_tau_smoke_study.py for how it is locked.
"""

from __future__ import annotations

import torch

from mol_fingerprint import tanimoto_similarity


class MolSimilarityConstraint:
    """c(x) = tau_mol - Tanimoto(x, seed_smiles) <= 0, i.e. threshold_type="min"
    in peptide's ConstraintFunction vocabulary: feasible means "at least this
    similar to the seed"."""

    def __init__(self, seed_smiles: str, tau_mol: float):
        self.seed_smiles = seed_smiles
        self.tau_mol = tau_mol

    def __call__(self, x_list: list[str]) -> torch.Tensor:
        """Input: a list of decoded candidate SMILES (already produced by the VAE
        decoder -- may include unparseable strings, which get the worst possible
        constraint value here rather than being silently dropped; an invalid
        molecule is handled by the oracle returning NaN for it, one layer up, not
        by this constraint pretending it's simply "very infeasible").
        Output: tensor of shape (len(x_list), 1), values <= 0 mean feasible."""
        c_vals = self.query_black_box(x_list)
        return (self.tau_mol - c_vals).unsqueeze(-1)

    def query_black_box(self, x_list: list[str]) -> torch.Tensor:
        sims = []
        for x in x_list:
            sim = tanimoto_similarity(x, self.seed_smiles)
            # An unparseable candidate cannot be a feasible neighbor of the seed by
            # definition; -inf similarity makes c(x) = tau_mol - (-inf) = +inf,
            # i.e. maximally infeasible, never accidentally satisfying c(x) <= 0.
            sims.append(sim if sim is not None else float("-inf"))
        # Stays on CPU always, matching peptide's own APEXSimilarityConstraint (no
        # device placement there either): LOLBOState's bookkeeping tensors
        # (train_x/y/z/c) are CPU by convention -- update_next() itself does
        # z_next_.detach().cpu() / y_next_.detach().cpu() before every torch.cat,
        # so train_c must already be CPU-resident to match. Individual methods that
        # need GPU compute (the VAE, the GP surrogate) cuda() their own
        # inputs/outputs at the point of use instead of the bookkeeping tensors
        # being cuda-resident throughout.
        return torch.tensor(sims, dtype=torch.float32)
