from torchmanager_core import torch, _raise, deprecated
from torchmanager_core.typing import Any, Callable, Dict, List, Optional
from torchmanager_core.view import warnings

from ..metrics import Metric


class Loss(Metric):
    """
    The main loss function

    * extends: `..metrics.Metric`
    * implements: `..callbacks.protocols.Weighted`
    * Loss tensor is stayed in memory until reset is called

    Define the loss function by overriding `forward` method:
    >>> from torchmanager import Manager
    >>> class SomeLoss(Loss):
    ...     def forward(self, input: Any, target: Any) -> torch.Tensor:
    ...         ...
    >>> loss_fn = SomeLoss()
    >>> manager = Manager(..., loss_fn=loss_fn)

    Or passing existing pytorch losses:
    >>> import torch
    >>> cross_entropy = Loss(torch.nn.CrossEntropyLoss(...))
    >>> manager = Manager(..., loss_fn=cross_entropy)

    - Properties:
        - weight: A `float` of coeffiency applied to current loss function
    """

    __weight: float

    @property
    def weight(self) -> float:
        return self.__weight

    @weight.setter
    def weight(self, w: float) -> None:
        assert w > 0, f"Weight must be a positive number, got {w}."
        self.__weight = w

    def __init__(self, loss_fn: Optional[Callable[[Any, Any], torch.Tensor]] = None, target: Optional[str] = None, weight: float = 1) -> None:
        """
        Constructor

        - Parameters:
            - loss_fn: An optional `Callable` function that accepts input or `y_pred` in `Any` kind and target or `y_true` in `Any` kind as inputs and gives a loss in `torch.Tensor`
            - target: An optional `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(loss_fn, target=target)
        self.weight = weight

    def __call__(self, input: Any, target: Any) -> torch.Tensor:
        m: torch.Tensor = super().__call__(input, target) * self.weight
        self._results[-1] *= self.weight
        return m

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        if isinstance(self._metric_fn, torch.nn.parallel.DataParallel):
            return super().forward(input, target).mean(dim=0)
        else:
            return super().forward(input, target)


class MultiLosses(Loss):
    """
    A loss with multiple losses sum together

    * extends: `.Loss`

    Passing lists of `Loss` to the `MultiLosses`:
    >>> from torchmanager import Manager
    >>> loss_1 = Loss(...)
    >>> loss_2 = Loss(...)
    >>> loss_fn = MultiLosses([loss_1, loss_2])
    >>> manager = Manager(..., loss_fn=loss_fn)

    - Properties:
        - losses: A `torch.nn.ModuleList` of loss metrics in `Metric`
    """

    __losses: torch.nn.ModuleList

    @property
    def losses(self) -> torch.nn.ModuleList:
        return self.__losses

    def __init__(self, losses: List[Loss], target: Optional[str] = None, weight: float = 1) -> None:
        super().__init__(target=target, weight=weight)
        self.__losses = torch.nn.ModuleList(losses)

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        # initilaize
        loss = 0

        # get all losses
        for fn in self.losses:
            assert isinstance(fn, Loss), _raise(TypeError(f"Function {fn} is not a Loss object."))
            l = fn(input, target)
            loss += l

        # return loss
        assert isinstance(loss, torch.Tensor), _raise(TypeError("The total loss is not a valid `torch.Tensor`."))
        return loss


@deprecated("v1.1.0", "v1.2.0")
class MultiOutputsLosses(Loss):
    """
    A loss with multiple losses for multiple outputs

    * extends: `.Loss`
    * [Depreciation Warning]: `MultiOutputsLosses` has been deprecated in v1.1.0 and will be removed in v1.2.0, use `MultiLosses` along with `target` parameter for each loss instead.

    - Properties:
        - losses: A `dict` of loss metrics in `Metric`
    """

    __losses: torch.nn.ModuleDict

    @property
    def losses(self) -> torch.nn.ModuleDict:
        return self.__losses

    def __init__(self, loss_fns: Dict[str, Loss]) -> None:
        super().__init__()
        assert len(loss_fns) > 0, "The loss dictionary should not be empty."
        self.__losses = torch.nn.ModuleDict(loss_fns)

    def forward(self, input: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        # initilaize
        loss = 0

        # loop for losses
        for k, fn in self.losses.items():
            assert isinstance(fn, Loss), _raise(TypeError(f"Function {fn} is not a Loss object."))
            l = fn(input[k], target[k])
            loss += l

        # return loss
        assert isinstance(loss, torch.Tensor), _raise(TypeError("The total loss is not a valid `torch.Tensor`."))
        return loss


class ParallelLoss(torch.nn.parallel.DataParallel):
    """
    A data parallel loss function

    * extends: `torch.nn.parallel.DataParallel`
    * implements: `torchmanager_core.protocols.Resulting`
    """

    module: Loss

    @property
    def _target(self) -> Optional[str]:
        return self.module._target

    @_target.setter
    def _target(self, t: Optional[str]) -> None:
        self.module._target = t

    @property
    def result(self) -> torch.Tensor:
        return torch.concat(self.module._results).mean()

    @property
    def results(self) -> torch.Tensor:
        return torch.concat(self.module._results)

    def __init__(self, module: Loss, device_ids: Optional[List[int]] = ..., output_device: Optional[torch.device] = ..., dim: int = ...) -> None:
        super().__init__(module, device_ids, output_device, dim)

    def reset(self) -> None:
        self.module.reset()


def loss(fn: Callable[[Any, Any], torch.Tensor]) -> Loss:
    """
    The loss wrapping function that wrap a function into a loss

    Use as a decorator:
    >>> import torch
    >>> @loss
    >>> def some_loss_fn(input: Any, target: Any) -> torch.Tensor:
    ...    return ...
    >>> manager = (..., loss_fn=some_loss_fn)
    """
    return Loss(fn)


def loss_fn(target: Optional[str] = None, weight: float = 1) -> Callable[[Callable[[Any, Any], torch.Tensor]], Loss]:
    """
    The loss wrapping function that wrap a function with target and weight given into a loss

    Use as a decorator:
    >>> import torch
    >>> @loss_fn(target='out', weight=0.5)
    >>> def some_loss_fn(input: Any, target: Any) -> torch.Tensor:
    ...    return ...
    >>> manager = (..., loss_fn=some_loss_fn)
    """

    def wrapped_loss_fn(fn_to_wrap: Callable[[Any, Any], torch.Tensor]) -> Loss:
        return Loss(fn_to_wrap, target=target, weight=weight)

    return wrapped_loss_fn
