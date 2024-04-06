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

    - Properties:
        - ce_lambda: A `float` of the weight of the cross entropy loss
        - dice_lambda: A `float` of the weight of the dice loss
    """
    __ce_lambda: float
    __dice_lambda: float
    _ce_lambda: float
    """Deprecated weight of cross entropy loss"""
    _dice_lambda: float
    """Deprecated weight of dice loss"""

    @property
    def ce_lambda(self) -> float:
        """The weight of the cross entropy loss in `float`"""
        return self.__ce_lambda

    @ce_lambda.setter
    def ce_lambda(self, value: float) -> None:
        assert value >= 0, _raise(ValueError(f"CE lambda must be a non-negative number, got {value}."))
        self.__ce_lambda = value

    @property
    def dice_lambda(self) -> float:
        """The weight of the dice loss in `float`"""
        return self.__dice_lambda

    @dice_lambda.setter
    def dice_lambda(self, value: float) -> None:
        assert value >= 0, _raise(ValueError(f"Dice lambda must be a non-negative number, got {value}."))
        self.__dice_lambda = value

    def __init__(self, *args, ce_lambda: float = 1, dice_lambda: float = 1, smooth: int = 1, target: Optional[str] = None, weight: float = 1, **kwargs) -> None:
        CrossEntropy.__init__(self, *args, target=target, weight=weight, **kwargs)
        Dice.__init__(self, smooth=smooth, target=target, weight=weight)
        self.ce_lambda = ce_lambda
        self.dice_lambda = dice_lambda

    def convert(self, from_version: Version) -> None:
        if from_version < Version("v1.3"):
            self.ce_lambda = self._ce_lambda
            self.dice_lambda = self._dice_lambda

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        dice = Dice.forward(self, input, target)
        ce = CrossEntropy.forward(self, input, target)
        return self.ce_lambda * ce + self.dice_lambda * dice


class FocalCrossEntropy(Loss):
    """
    The focal cross entropy loss

    * extends: `.loss.Loss`

    - Properties:
        - alpha: A `float` of alpha in focal cross entropy
        - calculate_average: A `bool` flag of if calculate average for the focal loss
        - gamma: A `float` of gamma in focal cross entropy
        - ignore_index: An `int` of Specified target value that is ignored
    """
    __alpha: float
    gamma: float
    calculate_average: bool
    ignore_index: int

    @property
    def alpha(self) -> float:
        return self.__alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        assert value >= 0, _raise(ValueError(f"Alpha must be a non-negative number, got {value}."))
        self.__alpha = value

    def __init__(self, alpha: float = 1, gamma: float = 1, calculate_average: bool = True, ignore_index: int = 255, **kwargs: Any):
        """
        Constructor

        - Parameters:
            - alpha: A `float` of alpha in focal cross entropy
            - gamma: A `float` of gamma in focal cross entropy
            - calculate_average: A `bool` flag of if calculate average for the focal loss
            - ignore_index: An `int` of Specified target value that is ignored
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.calculate_average = calculate_average
        self.gamma = gamma
        self.ignore_index = ignore_index

    def convert(self, from_version: Version) -> None:
        if from_version < Version("v1.3"):
            self.alpha = self._alpha
            self.gamma = self._gamma
            self.calculate_average = self._calculate_average
            self.ignore_index = self._ignore_index

    def forward(self, inputs: Any, targets: Any) -> torch.Tensor:
        # calculate loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss: torch.Tensor = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # calculate average
        if self.calculate_average:
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
    __t: Optional[float]
    _metric_fn: torch.nn.KLDivLoss
    _t: Optional[float]
    replace_nan: bool

    @property
    def log_target(self) -> bool:
        return self._metric_fn.log_target

    @property
    def softmax_temperature(self) -> Optional[float]:
        return self.__t
    
    @softmax_temperature.setter
    def softmax_temperature(self, value: Optional[float]) -> None:
        assert value is None or value > 0, _raise(ValueError(f"A given temperature must be a positive number, got {value}."))
        self.__t = value

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
        self.softmax_temperature = softmax_temperature
        self.replace_nan = replace_nan

    def convert(self, from_version: Version) -> None:
        if from_version < Version("v1.1"):
            self.replace_nan = False
            self.softmax_temperature = None
        elif from_version < Version("v1.3"):
            self.softmax_temperature = self._t

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # softmax input and target
        if self.softmax_temperature is not None:
            temperatured_input = input / self.softmax_temperature
            temperatured_target = target / self.softmax_temperature
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
        if self.softmax_temperature is not None:
            loss *= self.softmax_temperature ** 2
            
        # check nan
        if self.replace_nan:
            loss = loss.nan_to_num(0, posinf=0)

        return loss
