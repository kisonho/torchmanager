# import typing modules
from turtle import forward
from typing import Any, Callable, List, Optional

# import required modules
import torch
from torch.nn import functional as F

# import core modules
from .metrics import Metric

class Loss(Metric):
    """
    The main loss function

    * Could be use as a decorator of a function
    * Loss tensor is stayed in memory until reset is called
    """
    def __init__(self, loss_fn: Optional[Callable[[Any, Any], torch.Tensor]] = None) -> None:
        """
        Constructor

        - Parameters:
            - loss_fn: A `Callable` function that accepts input or `y_pred` in `Any` kind and target or `y_true` in `Any` kind as inputs and gives a loss in `torch.Tensor`
        """
        super().__init__(loss_fn)

    def reset(self) -> None:
        for t in self._results:
            t.detach()
        return super().reset()

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        return super().forward(input, target)

class CrossEntropy(Loss):
    """The cross entropy loss"""
    def __init__(self, *args, **kwargs) -> None:
        loss_fn = torch.nn.CrossEntropyLoss(*args, **kwargs)
        super().__init__(loss_fn)

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        assert self.__metric_fn is not None, "[Loss Error]: Crossentropy loss has not been initialized."
        return self.__metric_fn(input, target)

class FocalCrossEntropy(Loss):
    """The focal cross entropy loss"""
    _alpha: float
    _gamma: float
    _calculate_average: bool
    _ignore_index: int

    def __init__(self, alpha: float = 1, gamma: float = 0, calculate_average: bool = True, ignore_index: int = 255):
        super().__init__()
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

class MultiLosses(Loss):
    """
    A loss with multiple losses
    
    - Properties:
        - losses: A `list` of loss metrics in `Metric`
    """
    __losses: List[Metric]

    @property
    def losses(self) -> List[Metric]:
        return self.__losses

    def __init__(self, losses: List[Metric]) -> None:
        super().__init__(self.forward)
        self.__losses = losses

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        # initilaize
        loss: List[torch.Tensor] = []

        # get all losses
        for fn in self.losses:
            loss.append(fn.forward(input, target))

        # sum
        return torch.tensor(loss).sum()
