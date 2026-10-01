"""The mol domain's LatentSpaceObjective: SELFIES VAE (from the upstream molecular
LOL-BO vendor clone) + our EnsembleMolOracle + our Tanimoto-to-seed constraint.

Deliberately NOT based on optimization/mol/_upstream/lolbo/lolbo/molecule_objective.py
-- that file hardcodes version asserts for an old selfies/rdkit-pypi/molsets
install (`assert ... == '2.0.0'` etc, verified stale in Milestone 1) and its
query_oracle uses smiles_to_desired_scores/GUACAMOL_TASK_NAMES, the rejected
QED/penalized-logP/target-rediscovery formulations the spec explicitly forbids
reusing. Only its vae_decode/vae_forward *shape* is mirrored here, rewritten
against our own oracle and constraint.

No decode-repair heuristics (spec: "Do not port peptide's decode repair
heuristics" -- peptide's vae_decode rewrites invalid tokens and substitutes "AAA"
for empty decodes; there is no molecular analogue). An unsanitizable decode is
passed through as whatever raw string the VAE/SELFIES decoder produced; the
oracle (mol_oracle.py::decode_and_canonicalize) is solely responsible for turning
that into NaN. This class must never intercept that and paper over it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import selfies as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mol_lolbo.latent_space_objective import LatentSpaceObjective  # noqa: E402
from mol_lolbo.utils.mol_utils.selfies_vae.data import collate_fn  # noqa: E402
from mol_lolbo.utils.mol_utils.selfies_vae.model_positional_unbounded import (  # noqa: E402
    InfoTransformerVAE, SELFIESDataset,
)
from mol_constraint import MolSimilarityConstraint  # noqa: E402

DEFAULT_VAE_STATEDICT = (
    Path(__file__).resolve().parent / "_upstream" / "lolbo" / "lolbo" / "utils"
    / "mol_utils" / "selfies_vae" / "state_dict" / "SELFIES-VAE-state-dict.pt"
)
VAE_LATENT_DIM = 256  # bottleneck_size=2 * d_model=128, verified in Milestone 1


class MoleculeObjective(LatentSpaceObjective):
    """One instance per (target, seed) task, mirroring peptide's
    ApexConstrainedDiverseObjective being bound to one reference sequence."""

    def __init__(
        self,
        oracle,
        seed_smiles: str,
        tau_mol: float | None = None,
        path_to_vae_statedict: Path = DEFAULT_VAE_STATEDICT,
        max_string_length: int = 1024,
        xs_to_scores_dict: dict | None = None,
        num_calls: int = 0,
        task_id: str = "",
        use_inloop_constraint: bool = False,
    ):
        self.oracle = oracle  # exposes query_oracle(candidates, input_kind=...) -> list[float]
        self.seed_smiles = seed_smiles
        self.tau_mol = tau_mol
        self.path_to_vae_statedict = path_to_vae_statedict
        self.max_string_length = max_string_length
        self.dim = VAE_LATENT_DIM
        self.smiles_to_selfies: dict[str, str] = {}
        # Milestone 4 finding (MOL_LATENT_SPACE_FINDING.md): this VAE's latent
        # space does not preserve Tanimoto-to-seed even at the tightest radius
        # tested, so an in-loop GP constraint surrogate has no learnable structure
        # to fit -- default is unconstrained latent BO (pure potency), with
        # tau_mol enforced only at candidate-generation time (feasible-only init
        # pool, mol_init_candidates.py) and at evaluation time (post-hoc masking,
        # mirroring peptide's best_objective_at_k). use_inloop_constraint=True
        # restores the originally-specified GP-constrained-BO wiring for anyone
        # who wants to reproduce/re-verify that finding.
        self.use_inloop_constraint = use_inloop_constraint
        if use_inloop_constraint:
            if tau_mol is None:
                raise ValueError("tau_mol is required when use_inloop_constraint=True")
            self.constraint_functions = [MolSimilarityConstraint(seed_smiles, tau_mol)]
        else:
            self.constraint_functions = []

        super().__init__(
            xs_to_scores_dict=xs_to_scores_dict or {},
            num_calls=num_calls,
            task_id=task_id,
        )

    def initialize_vae(self) -> None:
        self.dataobj = SELFIESDataset()
        self.vae = InfoTransformerVAE(dataset=self.dataobj)
        state_dict = torch.load(self.path_to_vae_statedict, map_location="cpu")
        self.vae.load_state_dict(state_dict, strict=True)
        self.vae.eval()
        self.vae.max_string_length = self.max_string_length
        # The vendored constrained-BO engine (mol_lolbo/lolbo.py, turbo.py --
        # byte-identical ports of peptide's own generic extension) hardcodes
        # .cuda() throughout, exactly as peptide's own copy does, since production
        # peptide runs always have a GPU. The VAE must live on the same device
        # those tensors arrive on; the oracle stays on CPU regardless (Milestone 3:
        # negligible GPU speedup for these fingerprint/AAC-based DTI models).
        if torch.cuda.is_available():
            self.vae = self.vae.cuda()

    def vae_decode(self, z) -> list[str]:
        if isinstance(z, torch.Tensor) is False:
            z = torch.as_tensor(z, dtype=torch.float32)
        if next(self.vae.parameters()).is_cuda:
            z = z.cuda()
        z = z.reshape(-1, 2, 128)
        self.vae.eval()
        sample = self.vae.sample(z=z)
        decoded_selfies = [self.dataobj.decode(sample[i].tolist()) for i in range(sample.size(-2))]
        decoded_smiles = []
        for selfie in decoded_selfies:
            try:
                smile = sf.decoder(selfie)
            except Exception:
                smile = ""  # not repaired -- oracle turns this into NaN, full stop
            decoded_smiles.append(smile)
            self.smiles_to_selfies[smile] = selfie
        return decoded_smiles

    def query_oracle(self, x_list: list[str]) -> list[float]:
        """x_list is already-decoded SMILES (vae_decode's output), not SELFIES."""
        if not x_list:
            return []
        return self.oracle.query_oracle(x_list, input_kind="smiles")

    def vae_forward(self, xs_batch: list[str]):
        """xs_batch: a list of SMILES. Returns (z, vae_loss), mirroring the
        upstream MoleculeObjective.vae_forward shape exactly (only the oracle
        dependency was removed, not this method)."""
        encoded = []
        for smile in xs_batch:
            if smile in self.smiles_to_selfies:
                selfie = self.smiles_to_selfies[smile]
            else:
                selfie = sf.encoder(smile)
                self.smiles_to_selfies[smile] = selfie
            tokenized = self.dataobj.tokenize_selfies([selfie])[0]
            encoded.append(self.dataobj.encode(tokenized).unsqueeze(0))
        X = collate_fn(encoded)
        if next(self.vae.parameters()).is_cuda:
            X = X.cuda()
        out = self.vae(X)
        vae_loss, z = out["loss"], out["z"]
        z = z.reshape(-1, self.dim)
        return z, vae_loss

    def compute_constraints(self, xs_batch: list[str]) -> torch.Tensor | None:
        if not self.constraint_functions:
            return None
        all_cvals = [cfunc(xs_batch) for cfunc in self.constraint_functions]
        return torch.cat(all_cvals, dim=-1)
