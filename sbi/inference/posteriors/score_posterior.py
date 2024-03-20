# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.
from typing import Optional, Union
from functools import identity

import torch
import torch.nn as nn
from torch import Tensor, log
from torch.distributions import Distribution

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.potentials.score_based_potential import score_estimator_based_potential
from sbi.samplers.rejection.rejection import accept_reject_sample
from sbi.types import Shape
from sbi.utils import check_prior, within_support
from sbi.utils.torchutils import ensure_theta_batched

from sbi.neural_nets.vf_estimators.score_estimator import ScoreEstimator
from sbi.neural_nets.vf_estimators.score_based_distribution import ScoreDistribution


class ScorePosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x_o)$ with `log_prob()` and `sample()` methods, only
    applicable to SNPE.<br/><br/>
    Class to obtain a posterior for a diffusion model built with a score_estimator (a neural network to approximate the score).
    However, for bounded priors, we can have leakage: it puts non-zero
    mass in regions where the prior is zero. The `ScoreEstimatorPosterior` class also handles these cases.
    Specifically, this class offers the following functionality:<br/>
    - correct the calculation of the log probability such that it compensates for the
      leakage.<br/> ???
    - reject samples that lie outside of the prior bounds.<br/><br/>
    For now, this class can not be used in combination with SNLE or SNRE.
    """

    def __init__(
        self,
        score_estimator: ScoreEstimator,
        prior: Distribution,
        max_sampling_batch_size: int = 10_000,
        device: Optional[str] = None,
        x_shape: Optional[torch.Size] = None,
        enable_transform: bool = True,  # ???
    ):
        """
        Args:
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            posterior_estimator: The trained neural posterior.
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
        # Because `DirectPosterior` does not take the `potential_fn` as input, it
        # builds it itself. The `potential_fn` and `theta_transform` are used only for
        # obtaining the MAP.
        check_prior(prior)
        potential_fn, theta_transform = score_estimator_based_potential(
            score_estimator=score_estimator,
            prior=prior,
            x_0=None
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
        self._leakage_density_correction_factor = None

        self._purpose = """It samples from the diffusion model given the score_estimator and rejects samples that
            lie outside of the prior bounds."""


    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        max_sampling_batch_size: int = 10_000,
        sample_with: Optional[str] = None,
        show_progress_bars: bool = True,
    ) -> Tensor:
        r"""Return samples from posterior distribution $p(\theta|x)$.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            sample_with: This argument only exists to keep backward-compatibility with
                `sbi` v0.17.2 or older. If it is set, we instantly raise an error.
            show_progress_bars: Whether to show sampling progress monitor.
        """

        num_samples = torch.Size(sample_shape).numel()
        condition_shape = self.score_estimator._condition_shape
        x = self._x_else_default_x(x)

        try:
            x = x.reshape(*condition_shape)
        except RuntimeError as err:
            raise ValueError(
                f"Expected a single `x` which should broadcastable to shape \
                  {condition_shape}, but got {x.shape}. For batched eval \
                  see issue #990"
            ) from err

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

        proposal = ScoreDistribution(score_estimator=self.score_estimator, condition = x, sample_with = 'sde')
        samples = accept_reject_sample(
            proposal=proposal,  # type Union[nn.module, Distribution, NeuralPosterior]
            accept_reject_fn=lambda theta: within_support(self.prior, theta),
            num_samples=num_samples,
            show_progress_bars=show_progress_bars,
            max_sampling_batch_size=max_sampling_batch_size,
            proposal_sampling_kwargs={"condition": x},
            alternative_method="build_posterior(..., sample_with='mcmc')",
        )[0]

        return samples

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

        raise NotImplementedError
        x = self._x_else_default_x(x)
        condition_shape = self.posterior_estimator._condition_shape
        try:
            x = x.reshape(*condition_shape)
        except RuntimeError as err:
            raise ValueError(
                f"Expected a single `x` which should broadcastable to shape \
                  {condition_shape}, but got {x.shape}. For batched eval \
                  see issue #990"
            ) from err

        # TODO Train exited here, entered after sampling?
        self.posterior_estimator.eval()

        theta = ensure_theta_batched(torch.as_tensor(theta))

        with torch.set_grad_enabled(track_gradients):
            # Evaluate on device, move back to cpu for comparison with prior.
            unnorm_log_prob = self.posterior_estimator.log_prob(theta, condition=x)

            # Force probability to be zero outside prior support.
            in_prior_support = within_support(self.prior, theta)

            masked_log_prob = torch.where(
                in_prior_support,
                unnorm_log_prob,
                torch.tensor(float("-inf"), dtype=torch.float32, device=self._device),
            )

            if leakage_correction_params is None:
                leakage_correction_params = dict()  # use defaults
            log_factor = (
                log(self.leakage_correction(x=x, **leakage_correction_params))
                if norm_posterior
                else 0
            )

            return masked_log_prob - log_factor
    '''
    @torch.no_grad()
    def leakage_correction(
        self,
        x: Tensor,
        num_rejection_samples: int = 10_000,
        force_update: bool = False,
        show_progress_bars: bool = False,
        rejection_sampling_batch_size: int = 10_000,
    ) -> Tensor:
        r"""Return leakage correction factor for a leaky posterior density estimate.

        The factor is estimated from the acceptance probability during rejection
        sampling from the posterior.

        This is to avoid re-estimating the acceptance probability from scratch
        whenever `log_prob` is called and `norm_posterior=True`. Here, it
        is estimated only once for `self.default_x` and saved for later. We
        re-evaluate only whenever a new `x` is passed.

        Arguments:
            num_rejection_samples: Number of samples used to estimate correction factor.
            show_progress_bars: Whether to show a progress bar during sampling.
            rejection_sampling_batch_size: Batch size for rejection sampling.

        Returns:
            Saved or newly-estimated correction factor (as a scalar `Tensor`).
        """

        def acceptance_at(x: Tensor) -> Tensor:
            return accept_reject_sample(
                proposal=self.posterior_estimator,
                accept_reject_fn=lambda theta: within_support(self.prior, theta),
                num_samples=num_rejection_samples,
                show_progress_bars=show_progress_bars,
                sample_for_correction_factor=True,
                max_sampling_batch_size=rejection_sampling_batch_size,
                proposal_sampling_kwargs={"condition": x},
            )[1]

        # Check if the provided x matches the default x (short-circuit on identity).
        is_new_x = self.default_x is None or (
            x is not self.default_x and (x != self.default_x).any()
        )

        not_saved_at_default_x = self._leakage_density_correction_factor is None

        if is_new_x:  # Calculate at x; don't save.
            return acceptance_at(x)
        elif not_saved_at_default_x or force_update:  # Calculate at default_x; save.
            assert self.default_x is not None
            self._leakage_density_correction_factor = acceptance_at(self.default_x)

        return self._leakage_density_correction_factor  # type: ignore
    '''
    
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
    ) -> Tensor:
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
        raise NotImplementedError
        return super().map(
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

