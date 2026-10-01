"""Product-of-(Gaussian-process)-Experts ensemble surrogate (POGPE/SGPE,
Schilling et al. 2016, "Scalable hyperparameter optimization with products
of Gaussian process experts") for the GP-expert-transfer baseline
(peptide_experiment/gp_expert_transfer.py).

Combination formula (standard generalized product-of-experts / precision-
weighted robust-BCM combination -- the standard formula for this method
family; not verbatim-verified against Schilling et al. 2016 directly, since
that paper isn't available in this repo): for K experts with per-expert
posterior mean mu_k(x)/variance sigma_k(x)^2 (each already including that
expert's own likelihood/observation noise) and fixed weights beta_k,

    tau(x)   = sum_k beta_k / sigma_k(x)^2
    mu(x)    = (1/tau(x)) * sum_k beta_k * mu_k(x) / sigma_k(x)^2
    var(x)   = 1 / tau(x)

POGPE: beta_k = 1/K for all K pretrained experts. SGPE: same K pretrained
experts at beta_k=1/K, plus one extra "target" expert at beta=sum(betas)=1.0
("the independent GP for the target dataset carries the same weight as the
entire set of experts").

This combination is inherently pointwise -- it only ever uses each expert's
own marginal per-point variance, never cross-covariance between candidate
points. That's structural to the whole PoE/BCM method family, not a
limitation introduced here, and it's sufficient for this repo's actual
acquisition path (Thompson sampling via MaxPosteriorSampling -- turbo.py's
"ei" attempt always falls back to "ts" in practice), which only ever needs
marginal samples per candidate point.
"""

from __future__ import annotations

import gpytorch
import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from linear_operator.operators import DiagLinearOperator

from .ppgpr import GPModelDKL


class PoEGPModel(torch.nn.Module):
    def __init__(self, experts: list[GPModelDKL], weights: list[float]):
        super().__init__()
        assert len(experts) == len(weights) and len(experts) >= 1, (
            f"need at least one (expert, weight) pair, got {len(experts)} experts and {len(weights)} weights"
        )
        self.experts = torch.nn.ModuleList(experts)  # ModuleList so .eval()/.train()/.cuda() recurse
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.num_outputs = 1  # required by botorch's Model duck-typing, mirrors GPModelDKL's own attribute

    def posterior(self, X, output_indices=None, observation_noise=False, *args, **kwargs) -> GPyTorchPosterior:
        means, variances = [], []
        for expert in self.experts:
            expert.eval()
            expert.likelihood.eval()
            p = expert.posterior(X)
            means.append(p.mean.squeeze(-1))
            # Clamp guards a near-degenerate expert's variance from blowing up tau via division by ~0.
            variances.append(p.variance.squeeze(-1).clamp_min(1e-8))
        means = torch.stack(means, dim=0)  # (K, ..., N)
        variances = torch.stack(variances, dim=0)
        w = self.weights.to(means.device).view(-1, *([1] * (means.dim() - 1)))
        precisions = w / variances
        tau = precisions.sum(dim=0)
        mu = (precisions * means).sum(dim=0) / tau
        var = 1.0 / tau
        mvn = gpytorch.distributions.MultivariateNormal(mu, DiagLinearOperator(var))
        return GPyTorchPosterior(mvn)
