# BaseClass is a class for conditionalDistributions sample() and log_prob() methods, e.g torch.distributions.Distribution,
from sbi.neural_nets.vf_estimators.score_estimator import ScoreEstimator
import torch
from torch.distributions.normal import Normal
from zuko.utils import odeint
from torch import Tensor


class SBDistribution:
    """Wrapper around ScoreEstimator to have objects with sample function"""

    def __init__(self, score_estimator: ScoreEstimator, noise_distribution):
        super().__init__()
        self.score_estimator = score_estimator
        self.drift_fn = score_estimator.drift_fn
        self.diffusion_fn = score_estimator.diffusion_fn
        self.context_shape = score_estimator.context_shape
        self.step_size = 1000
        self.noise_distribution = Normal(
            loc=score_estimator.mean, scale=score_estimator.stddev
        )

    def log_prob(self, inputs, context):
        raise NotImplementedError()

    def sample_with_sde(self, num_samples, context):
        if context is None:
            raise ValueError("No context is passed to SBDistribution.sample.")
        else:
            theta = self.noise_distribution.sample(sample_shape=(num_samples,))
            delta_t = (
                1 / self.step_size
            )  # depends if we want to ode and sde term by step_size, probably right?

            for step in range(self.step_size):
                t = (step + 1) / self.step_size
                theta = (
                    theta
                    + (
                        self.drift_fn(theta, t=t)
                        - (self.diffusion_fn(theta, t=t)) ** 2
                        * self.score_estimator(theta, context=context, t=t)
                    )
                    * delta_t
                    + self.diffusion_fn(theta, t=t)
                    * torch.randn((num_samples,))*delta_t
                )

                return theta

    def sample_with_ode(self, num_samples, context):
        def f(theta: Tensor, t) -> Tensor:
            return self.drift_fn(theta, context=context, t=t) - 0.5*(
                self.diffusion_fn(theta, context=context, t=t)
            ) ** 2 * self.score_estimator(theta, context=context, t=t)

        theta0 = self.noise_distribution.sample(
            sample_shape=(num_samples,), context=context
        )
        theta1 = odeint(f, theta0, 0.0, 1.0)

        return theta1
