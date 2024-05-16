from torchmanager_core import functional as F, torch
from torchmanager_core.typing import Any

from .loss import Loss


class Dice(Loss):
    """
    The dice loss

    * extends: `.loss.Loss`
    """
    _dim: int
    _smooth: float
    _softmax_input: bool

    def __init__(self, dim: int = 1, smooth: float = 1e-6, *, softmax_input: bool = True, **kwargs: Any) -> None:
        """
        Constructor

        - Parameters:
            - dim: An `int` of class dimension
            - smooth: An `float` of smooth value to avoid dividing zero
            - softmax_input: A `bool` flag of if softmax the input
        """
        super().__init__(**kwargs)
        self._dim = dim
        self._smooth = smooth
        self._softmax_input = softmax_input

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # softmax activation
        num_classes = input.shape[self.dim]
        input = input.softmax(self.dim).view(-1) if self.softmax_input else input.sigmoid().view(-1)

        # one-hot encoding
        target = F.one_hot(target.view(-1), num_classes).view(-1) if num_classes > 1 else target.view(-1)

        # calculate dice
        intersection = input * target
        dice = 1 - (2 * intersection.sum() + self.smooth) / (input.sum() + target.sum() + self.smooth)
        return dice.mean()
