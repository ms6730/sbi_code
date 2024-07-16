import math
from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor

from sbi.samplers.score.predictors import Predictor

CORRECTORS = {}


def get_corrector(name: str, predictor: Predictor, **kwargs) -> "Corrector":
    return CORRECTORS[name](predictor, **kwargs)


def register_corrector(name: str) -> Callable:
    def decorator(predictor: Callable) -> Callable:
        CORRECTORS[name] = predictor
        return predictor

    return decorator


class Corrector(ABC):
    def __init__(
        self,
        predictor: Predictor,
    ):
        self.predictor = predictor
        self.score_fn = predictor.score_fn
        self.device = predictor.device

    def __call__(self, theta: Tensor, t0: Tensor) -> Tensor:
        return self.correct(theta, t0)

    @abstractmethod
    def correct(self, theta: Tensor, t0: Tensor) -> Tensor:
        pass


@register_corrector("langevin")
class LangevinCorrector(Corrector):
    def __init__(
        self,
        predictor: Predictor,
        step_size: float = 1e-4,
        num_steps: int = 5,
    ):
        super().__init__(predictor)
        self.step_size = step_size
        self.std = math.sqrt(2 * self.step_size)
        self.num_steps = num_steps

    @torch.compile(dynamic=True)
    def correct(self, theta: Tensor, t0: Tensor) -> Tensor:
        for _ in range(self.num_steps):
            score = self.score_fn(theta, t0)
            eps = self.std * torch.randn_like(theta, device=self.device)
            theta = theta + self.step_size * score + eps

        return theta


@register_corrector("gibbs")
class GibbsCorrector(Corrector):
    def __init__(self, predictor: Predictor, num_steps: int = 5):
        super().__init__(predictor)
        self.num_steps = num_steps

    def noise(self, theta: Tensor, t0: Tensor, t1: Tensor) -> Tensor:
        f = self.predictor.drift(theta, t0)
        g = self.predictor.diffusion(theta, t0)
        eps = torch.randn_like(theta, device=self.device)
        dt = t1 - t0
        dt_sqrt = torch.sqrt(dt)
        return theta + f * dt + g * eps * dt_sqrt

    @torch.compile(dynamic=True)
    def correct(self, theta: Tensor, t0: Tensor) -> Tensor:
        for _ in range(self.num_steps):
            theta = self.noise(theta, t0, t0 + 0.1)
            theta = self.predictor(theta, t0 + 0.1, t0)
        return theta
