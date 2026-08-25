"""BPO (Balanced Preference Optimization) loss -- an asymmetric alternative to
stock torchtune.rlhf.loss.DPOLoss, per "BPO: Revisiting Preference Modeling in
Direct Preference Optimization" (arxiv 2506.03557), Section 2.1 Eq. 5:

    r_w = log(pi_theta(y_w|x) / pi_ref(y_w|x))
    r_l = log(pi_theta(y_l|x) / pi_ref(y_l|x))
    logits = min(beta * r_w, -alpha * beta * r_l)
    loss = -log_sigmoid(logits)

alpha (the paper's "gap adaptor", in (0, 1], default 0.5) balances how much the
loss can be driven by improving the chosen side vs. suppressing the rejected
side: when r_w <= -alpha * r_l, the min is beta*r_w and the loss pushes to
improve the chosen response; otherwise the min is -alpha*beta*r_l and the loss
pushes to suppress the rejected response. This differs from stock DPOLoss,
which always optimizes the linear difference beta*(r_w - r_l) regardless of
which side is already well-separated.

Same forward() signature/return shape as torchtune.rlhf.loss.DPOLoss (drop-in
replacement via a config's loss._component_ override) -- no custom recipe
needed, since this only changes the loss formula on the same single
(chosen, rejected) batch the stock lora_dpo_distributed recipe already
produces.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1, alpha: float = 0.5):
        super().__init__()
        self.beta = beta
        self.alpha = alpha

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r_w = policy_chosen_logps - reference_chosen_logps
        r_l = policy_rejected_logps - reference_rejected_logps

        logits = torch.minimum(self.beta * r_w, -self.alpha * self.beta * r_l)
        losses = -F.logsigmoid(logits)

        # Same reward convention as stock DPOLoss (not alpha-scaled) so
        # chosen_rewards/rejected_rewards stay directly comparable in
        # tensorboard across DPOLoss- and BPOLoss-trained checkpoints.
        chosen_rewards = (self.beta * r_w).detach()
        rejected_rewards = (self.beta * r_l).detach()

        return losses, chosen_rewards, rejected_rewards
