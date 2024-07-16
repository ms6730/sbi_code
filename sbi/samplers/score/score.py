from typing import Optional, Union

import torch
from torch import Tensor
from tqdm.auto import tqdm

from sbi.inference.potentials.score_based_potential import ScoreBasedPotential
from sbi.samplers.score.correctors import Corrector, get_corrector
from sbi.samplers.score.predictors import Predictor, get_predictor


class Diffuser:
    predictor: Predictor
    corrector: Optional[Corrector]

    def __init__(
        self,
        score_based_potential: ScoreBasedPotential,
        predictor: Union[str, Predictor],
        corrector: Optional[Union[str, Corrector]] = None,
        init_std_scale: float = 1.0,
    ):
        self.set_predictor(predictor, score_based_potential)
        self.set_corrector(corrector)
        self.device = self.predictor.device

        # Extract time limits from the score function
        self.T_min = score_based_potential.score_estimator.T_min
        self.T_max = score_based_potential.score_estimator.T_max

        # Extract initial moments
        self.init_mean = score_based_potential.score_estimator.mean_T
        self.init_std = init_std_scale * score_based_potential.score_estimator.std_T

        # Extract relevant shapes from the score function
        self.input_shape = score_based_potential.score_estimator.input_shape
        self.condition_shape = score_based_potential.score_estimator.condition_shape
        condition_dim = len(self.condition_shape)
        self.batch_shape = score_based_potential.x_o.shape[:-condition_dim]

    def set_predictor(
        self,
        predictor: Union[str, Predictor],
        score_based_potential: ScoreBasedPotential,
    ):
        if isinstance(predictor, str):
            self.predictor = get_predictor(predictor, score_based_potential)
        else:
            self.predictor = predictor

    def set_corrector(self, corrector: Optional[Union[str, Corrector]]):
        if corrector is None:
            self.corrector = None
        elif isinstance(corrector, Corrector):
            self.corrector = corrector
        else:
            self.corrector = get_corrector(corrector, self.predictor)

    def initialize(self, num_samples: int) -> Tensor:
        num_batch = self.batch_shape.numel()
        eps = torch.randn(
            (num_batch, num_samples) + self.input_shape, device=self.device
        )
        mean, std, eps = torch.broadcast_tensors(self.init_mean, self.init_std, eps)
        return mean + std * eps

    @torch.no_grad()
    def run(self, num_samples: int, ts: Tensor, show_progress_bars: bool = True):
        theta = self.initialize(num_samples)
        pbar = tqdm(
            range(1, ts.numel()),
            disable=not show_progress_bars,
            desc=f"Drawing {num_samples} posterior samples",
        )
        for i in pbar:
            t1 = ts[i - 1]
            t0 = ts[i]
            theta = self.predictor(theta, t1, t0)
            if self.corrector is not None:
                theta = self.corrector(theta, t0)
        return theta
