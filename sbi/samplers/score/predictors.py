from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor

from sbi.inference.potentials.score_based_potential import ScoreBasedPotential

PREDICTORS = {}


def get_predictor(name: str, score_based_potential: ScoreBasedPotential) -> "Predictor":
    return PREDICTORS[name](score_based_potential)


def register_predictor(name: str) -> Callable:
    def decorator(predictor: "Predictor") -> "Predictor":
        PREDICTORS[name] = predictor
        return predictor

    return decorator


class Predictor(ABC):
    def __init__(
        self,
        score_fn: ScoreBasedPotential,
    ):
        self.score_fn = score_fn
        self.device = score_fn.device

        # Extract relevant functions from the score function
        self.drift = self.score_fn.score_estimator.drift_fn
        self.diffusion = self.score_fn.score_estimator.diffusion_fn

    def __call__(self, theta: Tensor, t1: Tensor, t0: Tensor) -> Tensor:
        return self.predict(theta, t1, t0)

    @abstractmethod
    def predict(self, theta: Tensor, t1: Tensor, t0: Tensor) -> Tensor:
        pass


@register_predictor("euler_maruyma")
class EulerMaruyama(Predictor):
    def __init__(
        self,
        score_fn: ScoreBasedPotential,
        eta: float = 1.0,
    ):
        super().__init__(score_fn)
        self.eta = eta

    @torch.compile(fullgraph=True, dynamic=True)
    def predict(self, theta: Tensor, t1: Tensor, t0: Tensor):
        dt = t1 - t0
        dt_sqrt = torch.sqrt(dt)
        f = self.drift(theta, t1)
        g = self.diffusion(theta, t1)
        score = self.score_fn(theta, t1)
        f_backward = f - (1 + self.eta**2) / 2 * g**2 * score
        g_backward = self.eta * g
        return theta - f_backward * dt + g_backward * torch.randn_like(theta) * dt_sqrt


class DDIM(Predictor):
    def __init__(
        self,
        score_fn: ScoreBasedPotential,
        std_bridge: Callable = lambda t: t,
        eta: float = 1.0,
    ):
        super().__init__(score_fn)
        self.alpha_fn = score_fn.score_estimator.mean_t_fn
        self.std_fn = score_fn.score_estimator.std_fn
        self.std_bridge = std_bridge
        self.eta = eta

    @torch.compile(fullgraph=True, dynamic=True)
    def predict(self, theta: Tensor, t1: Tensor, t0: Tensor) -> Tensor:
        # dt = t1 - t0
        # dt_sqrt = torch.sqrt(dt)
        alpha = self.alpha_fn(t1)
        std = self.std_fn(t1)
        alpha_new = self.alpha_fn(t0)
        std_new = self.std_fn(t0)
        std_bridge = self.std_bridge(t0)
        score = self.score_fn(theta, t0)
        eps_pred = -std * score

        x0_pred = (theta - std * eps_pred) / alpha
        bridge_correction = torch.sqrt(std_new**2 - std_bridge**2) * eps_pred
        bridge_noise = torch.randn_like(theta) * std_bridge

        new_position = alpha_new * x0_pred + bridge_correction + bridge_noise

        return new_position
