"""Feasibility-aware ORPT (FA-ORPT) loss.

Separates objective preference from feasibility preference in ORPT's
DPO-style ranking objective. Motivation (see imp_plan/04_orpt_future_plan.md's
2026-07-22 entry): the plain DPO-style ORPT loss (torchtune.rlhf.loss.DPOLoss,
used by ../torchtune_config/qwen_2_5_3B_lora_dpo.yaml) shows both chosen and
rejected u = log pi_theta - log pi_ref decreasing together during training,
with an insignificant margin -- it appears to win pairs by suppressing the
rejected sample rather than raising the chosen one. Separately,
experiments/constraint_violation/ established that naive objective-only
ranking lets ORPT propose similarity-infeasible sequences at a much higher
rate than BOLT.

FA-ORPT uses a candidate's feasibility label (see
../make_dpo_train_data_csv.py's --pairing-mode feasibility_aware, which is
the only pairing mode that produces both-infeasible pairs and per-side
feasibility labels) to apply one of three loss branches per pair:

- feasible vs feasible: preserve the objective ranking, nudge the winner up,
  but only lightly discourage the loser from falling far below the
  reference (a losing-but-feasible candidate is still useful for BO
  init/coverage later, so it shouldn't be punished as hard as an infeasible
  one).
- feasible vs infeasible: push the feasible candidate up, push the
  infeasible candidate down hard.
- infeasible vs infeasible: objective ranking carries no meaningful
  preference signal here, so just push both down (symmetric in the two
  sides -- this branch does not use which one is nominally "chosen").

Class imbalance across these three branches: on real BO trajectory data only
~5-10% of candidates are feasible, so even with
make_dpo_train_data_csv.py's feasibility_aware pairing sampling a roughly
even 1:1:1 mix of pair *types* pool-wide (see its DEFAULT_FEASIBILITY_AWARE_PAIR_TYPE_MIX),
individual mini-batches (batch_size=4 in ../torchtune_config/qwen_2_5_3B_lora_fa_orpt.yaml)
can still land 0 examples of a given branch just from small-sample variance.
normalize_by_branch=True (default) guards against this at the loss level: each
present branch's contribution is divided by its own in-batch count before
the recipe's `loss.mean()`, so a batch's gradient reflects each branch
present roughly equally instead of being dominated by whichever branch
happens to have the most examples in that particular batch.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeasibilityAwareORPTLoss(nn.Module):
    """Args mirror the task spec's notation directly:
    beta         -- temperature on the feasible-vs-feasible ranking term (same role as DPOLoss.beta).
    gamma_obj    -- feasible-vs-feasible ranking margin.
    gamma_plus   -- target the winning feasible candidate's u should rise above.
    gamma_keep   -- floor the losing feasible candidate's u shouldn't fall below.
    lambda_up    -- weight on the "winning feasible candidate should rise" term.
    lambda_keep  -- weight on the "losing feasible candidate shouldn't fall too far" term.
    gamma_f      -- target the feasible candidate's u should rise above, in a feasible-vs-infeasible pair.
    gamma_i      -- target below which an infeasible candidate's u should fall (used with a sign flip, see forward()).
    lambda_inf   -- weight on infeasible-candidate-suppression terms (both feasible-vs-infeasible and infeasible-vs-infeasible).
    normalize_by_branch -- if True (default), divide each branch's per-example
        loss by its own in-batch count before the recipe's loss.mean(), so a
        batch's gradient weighs each present branch equally regardless of how
        many of each type happen to land in that batch (see module docstring's
        "class imbalance" note). If False, use the literal per-example mean
        over the whole batch (every pair weighted equally instead of every
        present branch weighted equally).
    """

    def __init__(
        self,
        beta: float = 0.1,
        gamma_obj: float = 0.0,
        gamma_plus: float = 0.0,
        gamma_keep: float = -0.1,
        lambda_up: float = 0.2,
        lambda_keep: float = 0.2,
        gamma_f: float = 0.0,
        gamma_i: float = 0.5,
        lambda_inf: float = 1.0,
        normalize_by_branch: bool = True,
    ):
        super().__init__()
        self.beta = beta
        self.gamma_obj = gamma_obj
        self.gamma_plus = gamma_plus
        self.gamma_keep = gamma_keep
        self.lambda_up = lambda_up
        self.lambda_keep = lambda_keep
        self.gamma_f = gamma_f
        self.gamma_i = gamma_i
        self.lambda_inf = lambda_inf
        self.normalize_by_branch = normalize_by_branch

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        chosen_feasible: torch.Tensor,
        rejected_feasible: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """All input tensors have shape (batch,). chosen_feasible /
        rejected_feasible are 0/1 (any numeric dtype -- cast to bool here).

        Returns (losses, chosen_rewards, rejected_rewards), the same 3-tuple
        shape torchtune.rlhf.loss.DPOLoss returns, so recipe.py's reused
        `loss.mean()` / `(chosen_rewards > rejected_rewards).float()` lines
        need no changes.
        """
        u_plus = policy_chosen_logps - reference_chosen_logps
        u_minus = policy_rejected_logps - reference_rejected_logps

        chosen_feasible = chosen_feasible.bool()
        rejected_feasible = rejected_feasible.bool()

        mask_invalid = ~chosen_feasible & rejected_feasible
        if mask_invalid.any():
            raise ValueError(
                f"{int(mask_invalid.sum().item())} pair(s) have an infeasible chosen / feasible "
                "rejected sample. pairing_mode='feasibility_aware' should never produce this -- "
                "treat as a pair-construction bug (make_dpo_train_data_csv.py's "
                "_pick_chosen_rejected), not something to silently swap here."
            )

        mask_ff = chosen_feasible & rejected_feasible
        mask_fi = chosen_feasible & ~rejected_feasible
        mask_ii = ~chosen_feasible & ~rejected_feasible

        l_ff = (
            -F.logsigmoid(self.beta * (u_plus - u_minus - self.gamma_obj))
            + self.lambda_up * F.softplus(self.gamma_plus - u_plus)
            + self.lambda_keep * F.softplus(self.gamma_keep - u_minus)
        )
        l_fi = F.softplus(self.gamma_f - u_plus) + self.lambda_inf * F.softplus(self.gamma_i + u_minus)
        l_ii = self.lambda_inf * (F.softplus(self.gamma_i + u_plus) + F.softplus(self.gamma_i + u_minus))

        if self.normalize_by_branch:
            l_ff = l_ff / mask_ff.sum().clamp(min=1)
            l_fi = l_fi / mask_fi.sum().clamp(min=1)
            l_ii = l_ii / mask_ii.sum().clamp(min=1)

        losses = torch.zeros_like(u_plus)
        losses = torch.where(mask_ff, l_ff, losses)
        losses = torch.where(mask_fi, l_fi, losses)
        losses = torch.where(mask_ii, l_ii, losses)

        chosen_rewards = u_plus.detach()
        rejected_rewards = u_minus.detach()

        return losses, chosen_rewards, rejected_rewards
