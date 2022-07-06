from torchmanager_core import torch, _raise
from torchmanager_core.typing import Any, Callable, Dict, List, Optional
from torchmanager_core.view import warnings

from ..metrics import Metric

class Loss(Metric):
    """
    The main loss function

    * Could be use as a decorator of a function
    * Loss tensor is stayed in memory until reset is called
    """
    def __init__(self, loss_fn: Optional[Callable[[Any, Any], torch.Tensor]] = None, target: Optional[str] = None) -> None:
        """
        Constructor

        - Parameters:
            - loss_fn: An optional `Callable` function that accepts input or `y_pred` in `Any` kind and target or `y_true` in `Any` kind as inputs and gives a loss in `torch.Tensor`
            - target: An optional `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(loss_fn, target=target)

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

    def __init__(self, losses: List[Loss], target: Optional[str] = None) -> None:
        super().__init__(target=target)
        self.__losses = torch.nn.ModuleList(losses)

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        # initilaize
        loss = 0

        # get all losses
        for fn in self.losses:
            assert isinstance(fn, Loss), _raise(TypeError(f"Function {fn} is not a Loss object."))
            l = fn(input, target)
            loss += l
        return loss if isinstance(loss, torch.Tensor) else torch.tensor(loss, dtype=torch.float)

class MultiOutputsLosses(Loss):
    """
    A loss with multiple losses for multiple outputs

    - [Pending Depreciation Warning]: `MultiOutputsLosses` will be deprecated in v1.1.0, use `MultiLosses` along with `target` parameter for each loss instead.
    
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
        warnings.warn("`MultiOutputsLosses` will be deprecated in v1.1.0, use `MultiLosses` along with `target` parameter for each loss instead.", PendingDeprecationWarning)

    def forward(self, input: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        # initilaize
        loss = 0

        # loop for losses
        for k, fn in self.losses.items():
            assert isinstance(fn, Loss), _raise(TypeError(f"Function {fn} is not a Loss object."))
            l = fn(input[k], target[k])
            loss += l
        return loss if isinstance(loss, torch.Tensor) else torch.tensor(loss, dtype=torch.float)

def loss(fn: Callable[[Any, Any], torch.Tensor]) -> Loss:
    """
    The loss wrapping function that wrap a function into a loss

    * Use as a decorator
    """
    return Loss(fn)