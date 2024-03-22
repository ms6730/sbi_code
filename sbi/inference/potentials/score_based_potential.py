# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.potentials.base_potential import BasePotential
from sbi.neural_nets.vf_estimators.score_estimator import ScoreEstimator
from sbi.sbi_types import TorchTransform
from sbi.utils import mcmc_transform


def score_estimator_based_potential(
    score_estimator: ScoreEstimator,
    prior: Distribution,
    x_o: Optional[Tensor],
    x_o_shape: Optional[Tuple[int, ...]] = None,
    enable_transform: bool = False,
) -> Tuple[Callable, TorchTransform]:
    r"""Returns the potential function for score estimators.

    Args:
        score_estimator: The neural network modelling the score.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the score.
        x_o_shape: The shape of the observed data.
        enable_transform: Whether to enable transforms. Not supported yet.

    """
    device = str(next(score_estimator.parameters()).device)

    potential_fn = ScoreBasedPotential(
        score_estimator, prior, x_o, x_o_shape, device=device
    )

    assert (
        enable_transform is False
    ), "Transforms are not yet supported for score estimators."
    theta_transform = mcmc_transform(prior, device=device, enable_transform=False)

    return potential_fn, theta_transform


class ScoreBasedPotential(BasePotential):
    allow_iid_x = True  # type: ignore

    def __init__(
        self,
        score_estimator: ScoreEstimator,
        prior: Optional[Distribution],
        x_o: Optional[Tensor],
        x_o_shape: Optional[Tuple[int, ...]] = None,
        device: str = "cpu",
    ):
        r"""Returns the score function for score-based methods.

        Args:
            score_estimator: The neural network modelling the score.
            prior: The prior distribution.
            x_o: The observed data at which to evaluate the posterior.
            x_o_shape: The shape of the observed data.
            device: The device on which to evaluate the potential.
        """

        super().__init__(prior, x_o, device=device)
        self.score_estimator = score_estimator
        self.x_o_shape = x_o_shape

    def __call__(
        self, theta: Tensor, diffusion_time: Tensor, track_gradients: bool = True
    ) -> Tensor:
        r"""Returns the potential function for score-based methods.

        Args:
            theta: The parameters at which to evaluate the potential.
            diffusion_time: The diffusion time.
            track_gradients: Whether to track gradients.

        Returns:
            The potential function.
        """
        if self._x_o is None:
            raise ValueError(
                "No observed data x_o is available. Please reinitialize \
                the potential or manually set self._x_o."
            )

        # (batch, *event)[1:] == event?
        # If no, multiple iid observations are are present.
        if self.x_o.shape[1:] == self.x_o_shape:
            score_trial_sum = self.score_estimator.forward(
                input=theta, condition=self.x_o, times=diffusion_time
            )
        else:
            if self.prior is None:
                raise ValueError(
                    "No observed data prior is available. Please reinitialize \
                    the potential or manually set the prior."
                )
            if self.x_o_shape is None:
                raise ValueError(
                    "No observed data shape is available. Please reinitialize \
                    the potential or manually set the shape."
                )

            score_trial_sum = _bridge(
                x=self.x_o,
                x_shape=self.x_o_shape,
                theta=theta.to(self.device),
                estimator=self.score_estimator,
                diffusion_time=diffusion_time,
                prior=self.prior,
                track_gradients=track_gradients,
            )

        return score_trial_sum


def _bridge(
    x: Tensor,
    x_shape: Tuple[int, ...],
    theta: Tensor,
    estimator: ScoreEstimator,
    diffusion_time: Tensor,
    prior: Distribution,
    track_gradients: bool = False,
):
    r"""
    Returns the score-based potential for multiple IID observations. This can require a special solver
    to obtain the correct tall posterior.

    Args:
        x: The observed data.
        x_shape: The shape of the observed data.
        theta: The parameters at which to evaluate the potential.
        estimator: The neural network modelling the score.
        diffusion_time: The diffusion time.
        prior: The prior distribution.
        track_gradients: Whether to track gradients.
    """

    assert (
        next(estimator.parameters()).device == x.device and x.device == theta.device
    ), f"""device mismatch: estimator, x, theta: \
        {next(estimator.parameters()).device}, {x.device},
        {theta.device}."""

    # Get number of observations which are left from event_shape if they exist.
    num_obs = x.shape[-len(x_shape) - 1]

    # Calculate likelihood in one batch.
    # TODO needs to conform with the new shape handling
    with torch.set_grad_enabled(track_gradients):
        score_trial_batch = estimator.forward(
            input=theta, condition=x, times=diffusion_time
        )

        score_trial_sum = score_trial_batch.sum(0)

    return score_trial_sum + _get_prior_contribution(
        diffusion_time, prior, theta, num_obs
    )


def _get_prior_contribution(
    diffusion_time: Tensor,
    prior: Distribution,
    theta: Tensor,
    num_obs: int,
):
    r"""
    Returns the prior contribution for multiple IID observations.

    Args:
        diffusion_time: The diffusion time.
        prior: The prior distribution.
        theta: The parameter values at which to evaluate the prior contribution.
        num_obs: The number of independent observations.
    """
    # This method can be used to add several different bridges
    # to obtain the posterior for multiple IID observations.
    # For now, it only implements the approach by Geffner et al.

    # TODO Check if prior has the grad property else use torch autograd.
    # For now just use autograd.

    log_prob_theta = prior.log_prob(theta)

    grad_log_prob_theta = torch.autograd.grad(
        log_prob_theta,
        theta,
        grad_outputs=torch.ones_like(log_prob_theta),
        create_graph=True,
    )[0]

    return ((1 - num_obs) * (1.0 - diffusion_time)) * grad_log_prob_theta
