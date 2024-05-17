from torchmanager_core import functional as F, torch, _raise
from torchmanager_core.typing import Any
from torchmanager_core.version.version import Version

from .loss import Loss


class Dice(Loss):
    """
    The dice loss

    * extends: `.loss.Loss`
    """
    __dim: int
    __smooth: float
    _dim: int
    """Deprecated dimension property"""
    _smooth: float
    """Deprecated smooth property"""
    _softmax_input: bool
    """Deprecated softmax input property"""

    softmax_input: bool
    """A `bool` flag of if softmax the input"""

    @property
    def dim(self) -> int:
        """The dimension of the class in `int`"""
        return self.__dim

    @dim.setter
    def dim(self, value: int) -> None:
        assert value >= 0, _raise(ValueError(f"The dimension must be a positive number, got {value}."))
        self.__dim = value

    @property
    def smooth(self) -> float:
        """The smooth value in `float`"""
        return self.__smooth

    @smooth.setter
    def smooth(self, value: float) -> None:
        assert value >= 0, _raise(ValueError(f"The smooth value must be a non-negative number, got {value}."))
        self.__smooth = value

    def __init__(self, dim: int = 1, smooth: float = 1e-6, *, softmax_input: bool = True, **kwargs: Any) -> None:
        """
        Constructor

        - Parameters:
            - dim: An `int` of class dimension
            - smooth: An `float` of smooth value to avoid dividing zero
            - softmax_input: A `bool` flag of if softmax the input
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.smooth = smooth
        self.softmax_input = softmax_input

    def convert(self, from_version: Version) -> None:
        if from_version < Version("v1.3"):
            self.dim = self._dim
            self.smooth = self._smooth
            self.softmax_input = self._softmax_input

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
