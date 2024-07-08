# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.
from typing import Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.potentials.score_based_potential import (
    score_estimator_based_potential,
)
from sbi.neural_nets.estimators.score_estimator import ConditionalScoreEstimator
from sbi.samplers.score.score import score_based_sampler
from sbi.sbi_types import Shape
from sbi.utils import check_prior


class ScorePosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x_o)$ with `log_prob()` and `sample()` methods. It samples
    from the diffusion model given the score_estimator and rejects samples that lie
    outside of the prior bounds.

    The posterior is defined by a score estimator and a prior. The score estimator
    provides the gradient of the log-posterior with respect to the parameters. The
    prior is used to reject samples that lie outside of the prior bounds.

    NOTE: The `log_prob()` method is not implemented yet. It will be implemented in a
    future release using the probability flow ODEs.

    """

    def __init__(
        self,
        score_estimator: ConditionalScoreEstimator,
        prior: Distribution,
        max_sampling_batch_size: int = 10_000,
        device: Optional[str] = None,
        x_shape: Optional[torch.Size] = None,
        enable_transform: bool = True,  # ???
    ):
        """
        Args:
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            score_estimator: The trained neural score estimator.
            max_sampling_batch_size: Batchsize of samples being drawn from
                the proposal at every iteration.
            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
                `potential_fn.device` is used.
            x_shape: Shape of a single simulator output. If passed, it is used to check
                the shape of the observed data and give a descriptive error.
            enable_transform: Whether to transform parameters to unconstrained space
                during MAP optimization. When False, an identity transform will be
                returned for `theta_transform`.
        """

        check_prior(prior)
        potential_fn, theta_transform = score_estimator_based_potential(
            score_estimator=score_estimator,
            prior=prior,
            x_o=None,
            x_o_shape=x_shape,
        )

        super().__init__(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            device=device,
            x_shape=x_shape,
        )

        self.prior = prior
        self.score_estimator = score_estimator

        self.max_sampling_batch_size = max_sampling_batch_size

        self._purpose = """It samples from the diffusion model given the \
            score_estimator and rejects samples that
            lie outside of the prior bounds."""

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        method: str = "euler_maruyma",
        steps: int = 500,
        max_sampling_batch_size: int = 10_000,
        sample_with: Optional[str] = None,
        show_progress_bars: bool = True,
    ) -> Tensor:
        r"""Return samples from posterior distribution $p(\theta|x)$.

        Args:
            sample_shape: Shape of the samples to be drawn.
            x: Deprecated - use `.set_default_x()` prior to `.sample()`.
            method: Method to use for sampling. Currently, only "euler_maruyma" is
                supported.
            steps: Number of steps to take for the Euler-Maruyama method.
            max_sampling_batch_size: Maximum batch size for sampling.
            sample_with: Deprecated - use `.build_posterior(sample_with=...)` prior to
                `.sample()`.
            show_progress_bars: Whether to show a progress bar during sampling.
        """

        num_samples = torch.Size(sample_shape).numel()
        # condition_shape = self.score_estimator._condition_shape

        x = self._x_else_default_x(x)
        self.potential_fn.set_x(
            x.unsqueeze(0)
        )  # TODO Fix when new batching rules are in

        max_sampling_batch_size = (
            self.max_sampling_batch_size
            if max_sampling_batch_size is None
            else max_sampling_batch_size
        )

        if sample_with is not None:
            raise ValueError(
                f"You set `sample_with={sample_with}`. As of sbi v0.18.0, setting "
                f"`sample_with` is no longer supported. You have to rerun "
                f"`.build_posterior(sample_with={sample_with}).`"
            )

        # TODO: Prior can be `None` (extract from score estimator)
        theta_dim = self.prior.event_shape.numel()

        proposal = torch.distributions.Normal(
            torch.zeros(theta_dim), torch.ones(theta_dim)
        )  # TODO This must be extracted from the score estimator (can be different!)

        T_max = self.score_estimator.T_max
        T_min = self.score_estimator.T_min
        ts = torch.linspace(T_max, T_min, steps)

        # TODO: Use `method` to select the correct sampler
        samples = score_based_sampler(
            score_based_potential=self.potential_fn,  # type: ignore
            proposal=proposal,
            drift=self.score_estimator.drift_fn,
            diffusion=self.score_estimator.diffusion_fn,
            ts=ts,
            dim_theta=theta_dim,
            num_samples=num_samples,
        )

        # TODO: Implement accept-reject sampling ?
        # samples = accept_reject_sample(
        #     proposal=proposal,  # type Union[nn.module, Distribution, NeuralPosterior]
        #     accept_reject_fn=lambda theta: within_support(self.prior, theta),
        #     num_samples=num_samples,
        #     show_progress_bars=show_progress_bars,
        #     max_sampling_batch_size=max_sampling_batch_size,
        #     proposal_sampling_kwargs={"condition": x},
        #     alternative_method="build_posterior(..., sample_with='mcmc')",
        # )[0]

        return samples.reshape(sample_shape + self.prior.event_shape)

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
        norm_posterior: bool = True,
        track_gradients: bool = False,
        leakage_correction_params: Optional[dict] = None,
    ) -> Tensor:
        r"""Returns the log-probability of the posterior $p(\theta|x)$.

        Args:
            theta: Parameters $\theta$.
            norm_posterior: Whether to enforce a normalized posterior density.
                Renormalization of the posterior is useful when some
                probability falls out or leaks out of the prescribed prior support.
                The normalizing factor is calculated via rejection sampling, so if you
                need speedier but unnormalized log posterior estimates set here
                `norm_posterior=False`. The returned log posterior is set to
                -∞ outside of the prior support regardless of this setting.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.
            leakage_correction_params: A `dict` of keyword arguments to override the
                default values of `leakage_correction()`. Possible options are:
                `num_rejection_samples`, `force_update`, `show_progress_bars`, and
                `rejection_sampling_batch_size`.
                These parameters only have an effect if `norm_posterior=True`.

        Returns:
            `(len(θ),)`-shaped log posterior probability $\log p(\theta|x)$ for θ in the
            support of the prior, -∞ (corresponding to 0 probability) outside.
        """

        raise NotImplementedError("log_prob() is not implemented yet.")

    def sample_batched(
        self,
        sample_shape: torch.Size,
        x: Tensor,
        max_sampling_batch_size: int = 10000,
        show_progress_bars: bool = True,
    ) -> Tensor:
        raise NotImplementedError("Batched sampling is not implemented yet.")

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1_000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
        force_update: bool = False,
    ) -> Optional[Tensor]:
        r"""Returns the maximum-a-posteriori estimate (MAP).

        The method can be interrupted (Ctrl-C) when the user sees that the
        log-probability converges. The best estimate will be saved in `self._map` and
        can be accessed with `self.map()`. The MAP is obtained by running gradient
        ascent from a given number of starting positions (samples from the posterior
        with the highest log-probability). After the optimization is done, we select the
        parameter set that has the highest log-probability after the optimization.

        Warning: The default values used by this function are not well-tested. They
        might require hand-tuning for the problem at hand.

        For developers: if the prior is a `BoxUniform`, we carry out the optimization
        in unbounded space and transform the result back into bounded space.

        Args:
            x: Deprecated - use `.set_default_x()` prior to `.map()`.
            num_iter: Number of optimization steps that the algorithm takes
                to find the MAP.
            learning_rate: Learning rate of the optimizer.
            init_method: How to select the starting parameters for the optimization. If
                it is a string, it can be either [`posterior`, `prior`], which samples
                the respective distribution `num_init_samples` times. If it is a
                tensor, the tensor will be used as init locations.
            num_init_samples: Draw this number of samples from the posterior and
                evaluate the log-probability of all of them.
            num_to_optimize: From the drawn `num_init_samples`, use the
                `num_to_optimize` with highest log-probability as the initial points
                for the optimization.
            save_best_every: The best log-probability is computed, saved in the
                `map`-attribute, and printed every `save_best_every`-th iteration.
                Computing the best log-probability creates a significant overhead
                (thus, the default is `10`.)
            show_progress_bars: Whether to show a progressbar during sampling from the
                posterior.
            force_update: Whether to re-calculate the MAP when x is unchanged and
                have a cached value.
            log_prob_kwargs: Will be empty for SNLE and SNRE. Will contain
                {'norm_posterior': True} for SNPE.

        Returns:
            The MAP estimate.
        """
        super().map(
            x=x,
            num_iter=num_iter,
            num_to_optimize=num_to_optimize,
            learning_rate=learning_rate,
            init_method=init_method,
            num_init_samples=num_init_samples,
            save_best_every=save_best_every,
            show_progress_bars=show_progress_bars,
            force_update=force_update,
        )

    def _calculate_map(
        self,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1_000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
    ) -> Tensor:
        """Calculates the maximum-a-posteriori estimate (MAP).

        See `map()` method of child classes for docstring.
        """

        # if init_method == "posterior":
        #     inits = self.sample((num_init_samples,))
        # elif init_method == "proposal":
        #     inits = self.proposal.sample((num_init_samples,))  # type: ignore
        # elif isinstance(init_method, Tensor):
        #     inits = init_method
        # else:
        #     raise ValueError

        # TODO: Implement MAP optimization using the score estimator directly!

        raise NotImplementedError("MAP optimization is not implemented yet.")
