from torchmanager_core import functional as F, torch, Version, _raise
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
    _calculate_average: bool
    _gamma: float
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
    * A log-softmax will always be applied to `input`
    * A softmax will be applied to `target` only if `log_target` is `False`

    - Properties:
        - log_target: A `bool` flag of if `target` is in log space
        - replace_nan: A `boolean` flag of if replacing nan results to zeros
    """
    _metric_fn: torch.nn.KLDivLoss
    _t: Optional[float]
    replace_nan: bool

    @property
    def log_target(self) -> bool:
        return self._metric_fn.log_target

    def __init__(self, *args: Any, replace_nan: bool = False, softmax_temperature: Optional[float] = None, target: Optional[str] = None, weight: float = 1, **kwargs: Any) -> None:
        """
        Constructor

        - Parameters:
            - replace_nan: A `boolean` flag of if replacing nan results to zeros
            - softmax_temperature: An optional softmax temperature in `float`, softmax will not be applied if `None` is given.
            - target: An optional `str` of target name in `input` and `target` during direct calling
            - weight: A `float` of the loss weight
        """
        loss_fn = torch.nn.KLDivLoss(*args, **kwargs)
        super().__init__(loss_fn, target=target, weight=weight)
        self._t = softmax_temperature
        if self._t is not None:
            assert self._t > 0, _raise(ValueError(f"Temperature must be a positive number, got {self._t}."))
        self.replace_nan = replace_nan

    def convert(self, from_version: Version) -> None:
        if from_version < Version("v1.1"):
            self.replace_nan = False
            self._t = None

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # softmax input and target
        if self._t is not None:
            temperatured_input = input / self._t
            temperatured_target = target / self._t
            temperatured_input = temperatured_input.softmax(dim=1).log()
            temperatured_target = temperatured_target if self.log_target else target.softmax(dim=1)
        else:
            temperatured_input = input
            temperatured_target = target

        # check nan
        if self.replace_nan:
            input_max_value = torch.finfo(input.dtype).max
            target_max_value = torch.finfo(target.dtype).max
            temperatured_input = temperatured_input.nan_to_num(0, posinf=input_max_value, neginf=-1)
            temperatured_target = temperatured_target.nan_to_num(0, posinf=target_max_value, neginf=-1)

        # calculate kd-div loss
        loss = super().forward(temperatured_input, temperatured_target)
        
        # temperature control for knowledge distillation
        if self._t is not None:
            loss *= self._t ** 2
            
        # check nan
        if self.replace_nan:
            loss = loss.nan_to_num(0)

        return loss
