from torchmanager_core import functional as F, torch
from torchmanager_core.typing import Any

from .loss import Loss


class Dice(Loss):
    """
    The dice loss

    * extends: `.loss.Loss`
    """
    _smooth: int
    _softmax_input: bool

    def __init__(self, smooth: int = 1, softmax_input: bool = True, **kwargs: Any) -> None:
        """
        Constructor

        - Parameters:
            - smooth: An `int` of smooth value to avoid dividing zero
            - softmax_input: A `bool` flag of if softmax the input
        """
        super().__init__(**kwargs)
        self._smooth = smooth
        self._softmax_input = softmax_input

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # softmax activation
        input = F.softmax(input).view(-1) if self._softmax_input else input.view(-1)
        target = target.view(-1)

        # calculate dice
        intersection = input * target
        return (2 * intersection.sum() + self._smooth) / (input.sum() + target.sum() + self._smooth)
