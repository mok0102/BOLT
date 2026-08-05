"""Infeasible-sample suppression loss for the "dpo_ii" ablation.

Per-sample version of fa_orpt/loss.py's FeasibilityAwareORPTLoss l_ii term
(infeasible-vs-infeasible branch): l_ii = lambda_inf * (softplus(gamma_i+u_plus)
+ softplus(gamma_i+u_minus)) has no cross term between its two sides -- each
side's u = log pi_theta - log pi_ref is suppressed independently. So this
loss applies that same per-side formula to individually-sampled infeasible
completions (see ../make_infeasible_singles_csv.py), with no pairing at all.

Used alongside, not instead of, the stock torchtune.rlhf.loss.DPOLoss: this
ablation leaves standard ORPT's feasible-vs-feasible ranking loss completely
untouched (see dpo_ii/recipe.py), and just adds this term's mean on top.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfeasibleSuppressionLoss(nn.Module):
    def __init__(self, gamma_i: float = 0.5, lambda_inf: float = 1.0):
        super().__init__()
        self.gamma_i = gamma_i
        self.lambda_inf = lambda_inf

    def forward(self, policy_logps: torch.Tensor, reference_logps: torch.Tensor) -> torch.Tensor:
        """policy_logps/reference_logps: shape (batch,) per-sequence log
        probs (e.g. from torchtune.rlhf.get_batch_log_probs). Returns
        per-example losses, shape (batch,) -- caller takes .mean()."""
        u = policy_logps - reference_logps
        return self.lambda_inf * F.softplus(self.gamma_i + u)
