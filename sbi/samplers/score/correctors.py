import math
from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor

from sbi.inference.potentials.score_based_potential import ScoreBasedPotential

CORRECTORS = {}


def get_corrector(name: str) -> "Corrector":
    return CORRECTORS[name]


def register_corrector(name: str) -> Callable:
    def decorator(predictor: Callable) -> Callable:
        CORRECTORS[name] = predictor
        return predictor

    return decorator


class Corrector(ABC):
    def __init__(
        self,
        score_fn: ScoreBasedPotential,
    ):
        self.score_fn = score_fn
        self.device = score_fn.device

        # Extract relevant functions from the score function
        self.drift = self.score_fn.score_estimator.drift_fn
        self.diffusion = self.score_fn.score_estimator.diffusion_fn

    def __call__(self, theta: Tensor, t0: Tensor) -> Tensor:
        return self.correct(theta, t0)

    @abstractmethod
    def correct(self, theta: Tensor, t0: Tensor) -> Tensor:
        pass


@register_corrector("langevin")
class LangevinCorrector(Corrector):
    def __init__(
        self,
        score_fn: ScoreBasedPotential,
        step_size: float = 1e-4,
        num_steps: int = 5,
    ):
        super().__init__(score_fn)
        self.step_size = step_size
        self.std = math.sqrt(2 * self.step_size)
        self.num_steps = num_steps

    @torch.compile
    def correct(self, theta: Tensor, t0: Tensor) -> Tensor:
        for _ in range(self.num_steps):
            score = self.score_fn(theta, t0)
            eps = self.std * torch.randn_like(theta)
            theta = theta - self.step_size * score + eps

        return theta
