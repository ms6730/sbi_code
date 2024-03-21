from math import exp, log, sqrt
from typing import Tuple, Union, Optional, Callable

import torch
from torch import Tensor, nn

from sbi.neural_nets.vf_estimators.base import VectorFieldEstimator
from sbi.types import Shape


class ScoreEstimator(VectorFieldEstimator):
    r"""Score estimator for score-based generative models (e.g., denoising diffusion)."""

    def __init__(
        self,
        net: nn.Module,
        condition_shape: torch.Size,
        weight_fn: Union[str, Callable],
    ) -> None:
        """
        Generic score estimator class for SDEs.
        """        
        super().__init__(net, condition_shape)

        # Set lambdas (variance weights) function
        self._set_weight_fn(weight_fn)
        
        # Min time for diffusion (0 can be numerically unstable)
        self.T_min = 1e-3

        self.mean = (
            0.0  # this still needs to be computed (mean of the noise distribution)
        )
        self.std = 1.0  # same

    def mean_t_fn(self, times):
        raise NotImplementedError
    
    def mean_fn(self, x0, times):
        return self.mean_t_fn(times) * x0

    def std_fn(self, times):
        raise NotImplementedError

    def diffusion_fn(self, t):
        raise NotImplementedError

    def drift_fn(self, input, t):
        raise NotImplementedError

    def forward(self, input: Tensor, condition: Tensor, times: Tensor) -> Tensor:
        # Predict noise and divide by standard deviation to mirror target score.
        eps_pred = self.net([input, condition, times])
        std = self.std_fn(times)
        return eps_pred / std
    
    def loss(self, input: Tensor, condition: Tensor) -> Tensor:
        """Denoising score matching loss (Song et al., ICLR 2021)."""
        # Sample diffusion times.
        times = torch.clip(torch.rand((input.shape[0],)), self.T_min, 1.0)

        # Sample noise.
        eps = torch.randn_like(input)

        # Compute mean and standard deviation.
        mean = self.mean_fn(input, times)
        std = self.std_fn(times)

        # Get noised input, i.e., p(xt|x0).
        input_noised = mean + std * eps

        # Compute true score: -(mean - noised_input) / (std**2).
        score_target = -eps / std

        # Predict score.
        score_pred = self.forward(input_noised, condition, times)

        # Compute weights over time.
        weights = self.weight_fn(times)

        # Compute MSE loss between network output and true score.
        loss = torch.sum((score_target - score_pred)**2.0, axis=-1)        

        return weights*loss

    def _set_weight_fn(self, weight_fn):
        """Get the weight function."""
        if weight_fn == "identity":
            self.weight_fn = lambda t: 1
        elif weight_fn == "max_likelihood":
            self.weight_fn = lambda t: self.diffusion_fn(t)**2
        elif weight_fn == "variance":
            # From Song & Ermon, NeurIPS 2019.
            raise NotImplementedError
        elif callable(weight_fn):
            self.weight_fn = weight_fn
        else:
            raise ValueError(f"Weight function {weight_fn} not recognized.")


class VPScoreEstimator(ScoreEstimator):
    """Class for score estimators with variance preserving SDEs (i.e., DDPM)."""

    def __init__(
        self,
        net: nn.Module,
        condition_shape: torch.Size,
        weight_fn: Union[str, Callable] = "identity",
        beta_min: float = 0.1,
        beta_max: float = 20.0,
    ) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__(net, condition_shape, weight_fn=weight_fn)

    def mean_t_fn(self, times):
        a = torch.exp(
                -0.25 * times**2.0 * (self.beta_max - self.beta_min)
                - 0.5 * times * self.beta_min
            )
        return a.unsqueeze(-1)
    
    def std_fn(self, times):
        std =  1.0 - torch.exp(
            -0.5 * times**2.0 * (self.beta_max - self.beta_min)
            - times * self.beta_min
        )
        return torch.sqrt(std.unsqueeze(-1))

    def _beta_schedule(self, times):
        return self.beta_min + (self.beta_max - self.beta_min) * times

    def drift_fn(self, input, t):
        phi = -0.5 * self._beta_schedule(t)
        while len(phi.shape) < len(input.shape):
            phi = phi.unsqueeze(-1)
        return phi * input

    def diffusion_fn(self, input, t):
        g = torch.sqrt(
            self._beta_schedule(t)
        )
        while len(g.shape) < len(input.shape):
            g = g.unsqueeze(-1)
        return g


class subVPScoreEstimator(ScoreEstimator):
    """Class for score estimators with sub-variance preserving SDEs."""

    def __init__(
        self,
        net: nn.Module,
        condition_shape: torch.Size,
        weight_fn: Union[str, Callable] = "identity",
        beta_min: float = 0.1,
        beta_max: float = 20.0,
    ) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__(net, condition_shape, weight_fn=weight_fn)

    def mean_t_fn(self, times):
        a = torch.exp(
                -0.25 * times**2.0 * (self.beta_max - self.beta_min)
                - 0.5 * times * self.beta_min
        )                        
        return a.unsqueeze(-1)

    def std_fn(self, times):        
        std = 1.0 - torch.exp(
                -0.5 * times**2.0 * (self.beta_max - self.beta_min)
                - times * self.beta_min
            )        
        return std.unsqueeze(-1)

    def _beta_schedule(self, times):
        return self.beta_min + (self.beta_max - self.beta_min) * times

    def drift_fn(self, input, t):
        
        phi = -0.5 * self._beta_schedule(t) 
        
        while len(phi.shape) < len(input.shape):
            phi = phi.unsqueeze(-1)
        
        return phi * input

    def diffusion_fn(self, input, t):
        g = torch.sqrt(
            self._beta_schedule(t)
            * (-torch.exp(-2 * self.beta_min * t - (self.beta_max - self.beta_min) * t**2)))
        
        while len(g.shape) < len(input.shape):
            g = g.unsqueeze(-1)
        
        return g
        
        


class VEScoreEstimator(ScoreEstimator):
    """Class for score estimators with variance exploding SDEs (i.e., SMLD)."""

    def __init__(
        self,
        net: nn.Module,
        condition_shape: torch.Size,
        weight_fn: Union[str, Callable] = "identity",
        sigma_min: float = 0.01,
        sigma_max: float = 10.0,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        super().__init__(net, condition_shape, weight_fn=weight_fn)

    def mean_t_fn(self, times):
        return 1.0

    def std_fn(self, times):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** times
        return std.unsqueeze(-1)

    def _sigma_schedule(self, times):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** times

    def drift_fn(self, input, t):
        return torch.tenosr([0.0])

    def diffusion_fn(self, t):
        g = self._sigma_schedule(t) * torch.sqrt(2 * torch.log(self.sigma_max / self.sigma_min))
        
        while len(g.shape) < len(input.shape):
            g = g.unsqueeze(-1)
        
        return g
