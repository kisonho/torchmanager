from .core import functional as F, torch
from .core._typing import Any, Callable, Dict, List, Optional
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
        if isinstance(self._metric_fn, torch.nn.parallel.DataParallel):
            return super().forward(input, target).mean(dim=0)
        else: return super().forward(input, target)

class CrossEntropy(Loss):
    """The cross entropy loss"""
    def __init__(self, *args, **kwargs) -> None:
        loss_fn = torch.nn.CrossEntropyLoss(*args, **kwargs)
        super().__init__(loss_fn)

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        assert self._metric_fn is not None, "[Loss Error]: Crossentropy loss has not been initialized."
        return self._metric_fn(input, target)

class FocalCrossEntropy(Loss):
    """The focal cross entropy loss"""
    _alpha: float
    _gamma: float
    _calculate_average: bool
    _ignore_index: int

    def __init__(self, alpha: float = 1, gamma: float = 0, calculate_average: bool = True, ignore_index: int = 255):
        """
        Constructor

        - Parameters:
            - alpha: A `float` of alpha in focal cross entropy
            - gamma: A `float` of gamma in focal cross entropy
            - calculate_average: A `bool` flag of if calculate average for the focal loss
        """
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

class KLDiv(Loss):
    """KL-Div Loss"""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        loss_fn = torch.nn.KLDivLoss(*args, **kwargs)
        super().__init__(loss_fn)

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