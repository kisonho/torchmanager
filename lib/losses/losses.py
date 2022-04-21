from ..core import torch
from ..core.typing import Any, Callable, Dict, List, Optional
from ..metrics import Metric

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

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        if isinstance(self._metric_fn, torch.nn.parallel.DataParallel):
            return super().forward(input, target).mean(dim=0)
        else: return super().forward(input, target)

class MultiLosses(Loss):
    """
    A loss with multiple losses
    
    - Properties:
        - losses: A `list` of loss metrics in `Metric`
    """
    __losses: torch.nn.ModuleList

    @property
    def losses(self) -> torch.nn.ModuleList:
        return self.__losses

    def __init__(self, losses: List[Loss]) -> None:
        super().__init__(self.forward)
        self.__losses = torch.nn.ModuleList(losses)

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        # initilaize
        loss = torch.tensor(0, dtype=torch.float)

        # get all losses
        for fn in self.losses:
            assert isinstance(fn, Loss), f"[Runtime Error]: Function {fn} is not a Loss object."
            l = fn(input, target)
            loss = loss.to(l.device)
            loss += l
        return loss

class MultiOutputsLosses(Loss):
    """
    A loss with multiple losses for multiple outputs
    
    - Properties:
        - losses: A `dict` of loss metrics in `Metric`
    """
    __losses: torch.nn.ModuleDict

    @property
    def losses(self) -> torch.nn.ModuleDict:
        return self.__losses

    def __init__(self, loss_fns: Dict[str, Loss]) -> None:
        super().__init__()
        self.__losses = torch.nn.ModuleDict(loss_fns)

    def forward(self, input: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        # initilaize
        loss = torch.tensor(0, dtype=torch.float)

        # loop for losses
        for k, fn in self.losses.items():
            assert isinstance(fn, Loss), f"[Runtime Error]: Function {fn} is not a Loss object."
            l = fn(input[k], target[k])
            loss = loss.to(l.device)
            loss += l
        return loss

def loss(fn: Callable[[Any, Any], torch.Tensor]) -> Loss:
    """
    The loss wrapping function that wrap a function into a loss

    * Use as a decorator
    """
    return Loss(fn)