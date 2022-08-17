from torchmanager_core import functional as F, torch
from torchmanager_core.typing import Any, Optional

from .loss import Loss

class CrossEntropy(Loss):
    """
    The cross entropy loss
    
    * extends: `.loss.Loss`
    """
    def __init__(self, *args: Any, target: Optional[str] = None, weight: float = 1, **kwargs: Any) -> None:
        loss_fn = torch.nn.CrossEntropyLoss(*args, **kwargs)
        super().__init__(loss_fn, target=target, weight=weight)

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

class DiceCE(CrossEntropy, Dice):
    """
    Combined `Dice` loss and `CrossEntropy` loss
    
    * extends: `CrossEntropy`, `Dice`
    """
    _ce_lambda: float
    _dice_lambda: float

    def __init__(self, *args, ce_lambda: float = 1, dice_lambda: float = 1, smooth: int = 1, target: Optional[str] = None, weight: float = 1, **kwargs) -> None:
        CrossEntropy.__init__(self, *args, target=target, weight=weight, **kwargs)
        Dice.__init__(self, smooth=smooth, target=target, weight=weight)
        self._ce_lambda = ce_lambda
        self._dice_lambda = dice_lambda

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        dice = Dice.forward(self, input, target)
        ce = CrossEntropy.forward(self, input, target)
        return self._ce_lambda * ce + self._dice_lambda * dice

class FocalCrossEntropy(Loss):
    """
    The focal cross entropy loss
    
    * extends: `.loss.Loss`
    """
    _alpha: float
    _gamma: float
    _calculate_average: bool
    _ignore_index: int

    def __init__(self, alpha: float = 1, gamma: float = 0, calculate_average: bool = True, ignore_index: int = 255, **kwargs: Any):
        """
        Constructor

        - Parameters:
            - alpha: A `float` of alpha in focal cross entropy
            - gamma: A `float` of gamma in focal cross entropy
            - calculate_average: A `bool` flag of if calculate average for the focal loss
            - ignore_index: An `int` of Specified target value that is ignored
        """
        super().__init__(**kwargs)
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
    """
    KL-Div Loss
    
    * extends: `.loss.Loss`
    """
    def __init__(self, *args: Any, target: Optional[str] = None, weight: float = 1, **kwargs: Any) -> None:
        loss_fn = torch.nn.KLDivLoss(*args, **kwargs)
        super().__init__(loss_fn, target=target, weight=weight)