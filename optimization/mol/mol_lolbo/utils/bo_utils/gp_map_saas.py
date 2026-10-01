import gpytorch
import torch
from botorch.exceptions import UnsupportedError
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.constraints import Interval
from gpytorch.kernels import AdditiveKernel, Kernel, MaternKernel, ScaleKernel
from gpytorch.models import ApproximateGP
from gpytorch.priors import HalfCauchyPrior
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch.distributions.half_cauchy import HalfCauchy
from torch.nn import Parameter


class MAPSaasGPModel(ApproximateGP):
    def __init__(
        self,
        inducing_points,
        likelihood,
        num_taus=4,
    ):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        aug_batch_shape = inducing_points.shape[
            :-2
        ]  # to account for multiple batch dims

        self.covar_module = get_additive_map_saas_covar_module(
            ard_num_dims=inducing_points.shape[-1],
            num_taus=num_taus,
            batch_shape=aug_batch_shape,
        )

        self.num_outputs = 1
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)  # type: ignore

    def posterior(self, X, *args, **kwargs) -> GPyTorchPosterior:
        self.eval()  # make sure model is in eval mode
        self.likelihood.eval()
        dist = self.likelihood(self(X))

        return GPyTorchPosterior(dist)

    @torch.no_grad()
    def get_lengthscales(self):
        return torch.vstack(
            [k.base_kernel.lengthscale for k in self.covar_module.kernels]
        )


class LogTransformedInterval(Interval):
    """Modification of the GPyTorch interval class.

    The Interval class in GPyTorch will map the parameter to the range [0, 1] before
    applying the inverse transform. We don't want to do this when using log as an
    inverse transform. This class will skip this step and apply the log transform
    directly to the parameter values so we can optimize log(parameter) under the bound
    constraints log(lower) <= log(parameter) <= log(upper).
    """

    def __init__(self, lower_bound, upper_bound, initial_value=None):
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            transform=torch.exp,
            inv_transform=torch.log,
            initial_value=initial_value,
        )

        # Save the untransformed initial value
        self.register_buffer(
            "initial_value_untransformed",
            torch.tensor(initial_value).to(self.lower_bound)
            if initial_value is not None
            else None,
        )

    def transform(self, tensor):
        if not self.enforced:
            return tensor

        transformed_tensor = self._transform(tensor)
        return transformed_tensor

    def inverse_transform(self, transformed_tensor):
        if not self.enforced:
            return transformed_tensor

        tensor = self._inv_transform(transformed_tensor)
        return tensor


class SaasPriorHelper:
    """Helper class for specifying parameter and setting closures."""

    def __init__(self, tau: float | None = None):
        self._tau = torch.as_tensor(tau) if tau is not None else None

    def tau(self, m):
        return (
            self._tau.to(m.lengthscale)
            if self._tau is not None
            else m.raw_tau_constraint.transform(m.raw_tau)
        )

    def inv_lengthscale_prior_param_or_closure(self, m):
        return self.tau(m).unsqueeze(-1) / (m.lengthscale**2)

    def inv_lengthscale_prior_setting_closure(self, m, value):
        lb = m.raw_lengthscale_constraint.lower_bound
        ub = m.raw_lengthscale_constraint.upper_bound
        m._set_lengthscale((self.tau(m).unsqueeze(-1) / value).sqrt().clamp(lb, ub))

    def tau_prior_param_or_closure(self, m):
        return m.raw_tau_constraint.transform(m.raw_tau)

    def tau_prior_setting_closure(self, m, value):
        lb = m.raw_tau_constraint.lower_bound
        ub = m.raw_tau_constraint.upper_bound
        m.raw_tau.data.fill_(
            m.raw_tau_constraint.inverse_transform(value.clamp(lb, ub)).item()
        )


def add_saas_prior(
    base_kernel: Kernel, tau: float | None = None, log_scale: bool = True, **tkwargs
) -> Kernel:
    """Add a SAAS prior to a given base_kernel.

    The SAAS prior is given by tau / lengthscale^2 ~ HC(1.0). If tau is None,
    we place an additional HC(0.1) prior on tau similar to the original SAAS prior
    that relies on inference with NUTS.

    Args:
        base_kernel: Base kernel that has a lengthscale and uses ARD.
            Note that this function modifies the kernel object in place.
        tau: Value of the global shrinkage. If `None`, infer the global
            shrinkage parameter.
        log_scale: Set to `True` if the lengthscale and tau should be optimized on
            a log-scale without any domain rescaling. That is, we will learn
            `raw_lengthscale := log(lengthscale)` and this hyperparameter needs to
            satisfy the corresponding bound constraints. Setting this to `True` will
            generally improve the numerical stability, but requires an optimizer that
            can handle bound constraints, e.g., L-BFGS-B.

    Returns:
        Base kernel with SAAS priors added.

    Example:
        >>> matern_kernel = MaternKernel(...)
        >>> add_saas_prior(matern_kernel, tau=None)  # Add a SAAS prior
    """
    if not base_kernel.has_lengthscale:
        raise UnsupportedError("base_kernel must have lengthscale(s)")
    if hasattr(base_kernel, "lengthscale_prior"):
        raise UnsupportedError("base_kernel must not specify a lengthscale prior")

    batch_shape = base_kernel.raw_lengthscale.shape[:-2]
    IntervalClass = LogTransformedInterval if log_scale else Interval
    base_kernel.register_constraint(
        param_name="raw_lengthscale",
        constraint=IntervalClass(0.01, 1e4, initial_value=1),
        replace=True,
    )
    prior_helper = SaasPriorHelper(tau=tau)
    if tau is None:  # Place a HC(0.1) prior on tau
        base_kernel.register_parameter(
            name="raw_tau",
            parameter=Parameter(torch.full(batch_shape, 0.1, **tkwargs)),
        )
        base_kernel.register_constraint(
            param_name="raw_tau",
            constraint=IntervalClass(1e-3, 10, initial_value=0.1),
            replace=True,
        )
        base_kernel.register_prior(
            name="tau_prior",
            prior=HalfCauchyPrior(torch.tensor(0.1, **tkwargs)),
            param_or_closure=prior_helper.tau_prior_param_or_closure,
            setting_closure=prior_helper.tau_prior_setting_closure,
        )
    # Place a HC(1) prior on tau / lengthscale^2
    base_kernel.register_prior(
        name="inv_lengthscale_prior",
        prior=HalfCauchyPrior(torch.tensor(1.0, **tkwargs)),
        param_or_closure=prior_helper.inv_lengthscale_prior_param_or_closure,
        setting_closure=prior_helper.inv_lengthscale_prior_setting_closure,
    )
    return base_kernel


def get_additive_map_saas_covar_module(
    ard_num_dims: int, num_taus: int = 4, batch_shape: torch.Size | None = None
):
    """Return an additive map SAAS covar module.

    The constructed kernel is an additive kernel with `num_taus` terms. Each term is a
    scaled Matern kernel with a SAAS prior and a tau sampled from a HalfCauchy(0, 1)
    distrbution.

    Args:
        ard_num_dims: The number of inputs dimensions.
        num_taus: The number of taus to use (4 if omitted).
        batch_shape: Batch shape for the covar module.

    Returns:
        An additive MAP SAAS covar module.
    """
    batch_shape = batch_shape or torch.Size()
    kernels = []
    taus = [1, 0.1, 0.01, 0.001]
    # for _ in range(num_taus):
    for tau in taus:
        base_kernel = MaternKernel(
            nu=2.5, ard_num_dims=ard_num_dims, batch_shape=batch_shape
        )
        add_saas_prior(base_kernel=base_kernel, tau=tau)
        scaled_kernel = ScaleKernel(
            base_kernel=base_kernel,
            outputscale_constraint=LogTransformedInterval(1e-2, 1e4, initial_value=10),
            batch_shape=batch_shape,
        )
        kernels.append(scaled_kernel)
    return AdditiveKernel(*kernels)
