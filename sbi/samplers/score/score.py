
from typing import Callable, Any
from tqdm.auto import tqdm
import torch


def score_based_sampler(
    score_based_potential: Callable,
    proposal: Any,
    drift: Callable,
    diffusion: Callable,
    ts: torch.Tensor,
    num_samples: int = 1,
    show_progress_bars: bool = True,
):
    r"""Returns a sampler for score-based methods.

    Args:
        score_based_potential: The score-based potential function.
        proposal: The proposal distribution.
        drift: The drift function of the SDE.
        diffusion: The diffusion function of the SDE.

    Returns:
        A sampler for score-based methods.
    """


    pbar = tqdm(
        ts,
        disable=not show_progress_bars,
        desc=f"Drawing {num_samples} posterior samples",
    )

    for t in pbar:
        pass
            
            

    return theta
