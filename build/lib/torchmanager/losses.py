# import typing modules
from typing import Any, Callable, List

# import required modules
import torch

# import core modules
from .metrics import Metric

class Loss(Metric):
    '''The main loss function'''
    def __init__(self, loss_fn: Callable[[Any, Any], torch.Tensor]) -> None:
        '''
        Constructor

        - Parameters:
            - loss_fn: A `Callable` function that accepts input or `y_pred` in `Any` kind and target or `y_true` in `Any` kind as inputs and gives a loss in `torch.Tensor`
        '''
        super().__init__(loss_fn)

    def call(self, input: Any, target: Any) -> torch.Tensor:
        return super().call(input, target)

class CrossEntropy(Loss):
    '''The cross entropy loss'''
    def __init__(self, *args, **kwargs) -> None:
        loss_fn = torch.nn.CrossEntropyLoss(*args, **kwargs)
        super().__init__(loss_fn)

class MultiLosses(Loss):
    '''
    A loss with multiple losses
    
    - Properties:
        - losses: A `list` of loss metrics in `Metric`
    '''
    __losses: List[Metric]

    @property
    def losses(self) -> List[Metric]:
        return self.__losses

    def __init__(self, losses: List[Metric]) -> None:
        super().__init__(self.call)
        self.__losses = losses

    def call(self, input: Any, target: Any) -> torch.Tensor:
        # initilaize
        loss: List[torch.Tensor] = []

        # get all losses
        for fn in self.losses:
            loss.append(fn.call(input, target))

        # sum
        return torch.tensor(loss).sum()