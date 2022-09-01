from torchmanager_core import functional as F, torch
from torchmanager_core.typing import Any, Optional

from .dice import Dice
from .loss import Loss

class CrossEntropy(Loss):
    """
    The cross entropy loss
    
    * extends: `.loss.Loss`
    """
    def __init__(self, *args: Any, target: Optional[str] = None, weight: float = 1, **kwargs: Any) -> None:
        loss_fn = torch.nn.CrossEntropyLoss(*args, **kwargs)
        super().__init__(loss_fn, target=target, weight=weight)

class DiceCE(CrossEntropy, Dice):
    """
    Combined `Dice` loss and `CrossEntropy` loss
    
    * extends: `CrossEntropy`, `Dice`
    """
    __ce_lambda: float
    __dice_lambda: float

    @property
    def _ce_lambda(self) -> float: return self.__ce_lambda

    @property
    def _dice_lambda(self) -> float: return self.__dice_lambda

    def __init__(self, *args, ce_lambda: float = 1, dice_lambda: float = 1, smooth: int = 1, target: Optional[str] = None, weight: float = 1, **kwargs) -> None:
        CrossEntropy.__init__(self, *args, target=target, weight=weight, **kwargs)
        Dice.__init__(self, smooth=smooth, target=target, weight=weight)
        self.__ce_lambda = ce_lambda
        self.__dice_lambda = dice_lambda

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        dice = Dice.forward(self, input, target)
        ce = CrossEntropy.forward(self, input, target)
        return self._ce_lambda * ce + self._dice_lambda * dice

class FocalCrossEntropy(Loss):
    """
    The focal cross entropy loss
    
    * extends: `.loss.Loss`
    """
    __alpha: float
    __calculate_average: bool
    __gamma: float
    __ignore_index: int

    @property
    def _alpha(self) -> float: return self.__alpha

    @property
    def _calculate_average(self) -> bool: return self.__calculate_average

    @property
    def _gamma(self) -> float: return self.__gamma

    @property
    def _ignore_index(self) -> int: return self.__ignore_index

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
        self.__alpha = alpha
        self.__gamma = gamma
        self.__ignore_index = ignore_index
        self.__calculate_average = calculate_average

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