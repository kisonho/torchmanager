import torch
from torch import rand, rand_like, randint, randint_like, randn, randn_like, randperm
from typing import Sequence


def randbool(size: Sequence[int], /, p: float = 0.5) -> torch.Tensor:
    """
    Samples a `torch.Tensor` with its shape as `size` filled with random booleans. The sampled tensor will have true values at possibility `p`.

    - Parameters:
        - size: A `Sequence` of the output tensor shape in `int`
        - p: A possibility in `float`
    - Returns: A sampled `torch.Tensor`
    """
    # check possibility value
    if p <= 0 or p >= 1:
        raise ValueError(f"Possibility value must be in range (0, 1), got {p}.")

    # sampling
    return rand(size) < p


def randbool_like(t: torch.Tensor, /, p: float = 0.5) -> torch.Tensor:
    return randbool(t.shape, p=p)
