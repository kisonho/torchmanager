from torchmanager_core import functional as F, torch
from torchmanager_core.typing import Any, Optional

from .loss import Loss

class CrossEntropy(Loss):
    """The cross entropy loss"""
    def __init__(self, *args, target: Optional[str] = None, **kwargs) -> None:
        loss_fn = torch.nn.CrossEntropyLoss(*args, **kwargs)
        super().__init__(loss_fn, target=target)

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        assert self._metric_fn is not None, "[Loss Error]: Crossentropy loss has not been initialized."
        return self._metric_fn(input, target)

class Dice(Loss):
    """The dice loss"""
    _smooth: int
    _softmax_input: bool

    def __init__(self, smooth: int = 1, softmax_input: bool = True, target: Optional[str] = None) -> None:
        """
        Constructor

        - Parameters:
            - smooth: An `int` of smooth value to avoid dividing zero
            - softmax_input: A `bool` flag of if softmax the input
            - target: An optional `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(target=target)
        self._smooth = smooth
        self._softmax_input = softmax_input

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # softmax activation
        input = F.softmax(input).view(-1) if self._softmax_input else input.view(-1)
        target = target.view(-1)

        # calculate dice
        intersection = input * target
        return (2 * intersection.sum() + self._smooth) / (input.sum() + target.sum() + self._smooth)

class DiceCE(CrossEntropy, Dice):
    """Combined `Dice` loss and `CrossEntropy` loss"""
    _ce_lambda: float
    _dice_lambda: float

    def __init__(self, *args, ce_lambda: float = 1, dice_lambda: float = 1, smooth: int = 1, target: Optional[str] = None, **kwargs) -> None:
        CrossEntropy.__init__(self, *args, target=target, **kwargs)
        Dice.__init__(self, smooth=smooth, target=target)
        self._ce_lambda = ce_lambda
        self._dice_lambda = dice_lambda

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        dice = Dice.forward(self, input, target)
        ce = CrossEntropy.forward(self, input, target)
        return self._ce_lambda * ce + self._dice_lambda * dice

class FocalCrossEntropy(Loss):
    """The focal cross entropy loss"""
    _alpha: float
    _gamma: float
    _calculate_average: bool
    _ignore_index: int

    def __init__(self, alpha: float = 1, gamma: float = 0, calculate_average: bool = True, ignore_index: int = 255, target: Optional[str] = None):
        """
        Constructor

        - Parameters:
            - alpha: A `float` of alpha in focal cross entropy
            - gamma: A `float` of gamma in focal cross entropy
            - calculate_average: A `bool` flag of if calculate average for the focal loss
        """
        super().__init__(target=target)
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_index = ignore_index
        self._calculate_average = calculate_average

    def forward(self, inputs: Any, targets: Any) -> torch.Tensor:
        # calculate loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self._ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss: torch.Tensor = self._alpha * (1 - pt) ** self._gamma * ce_loss

        # calculate average
        if self._calculate_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class KLDiv(Loss):
    """KL-Div Loss"""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        loss_fn = torch.nn.KLDivLoss(*args, **kwargs)
        super().__init__(loss_fn)