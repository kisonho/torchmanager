from itertools import chain
from torchmanager_core import abc, devices, torch, _raise
from torchmanager_core.typing import Any, Callable, Generic, TypeVar, cast

from .protocols import BaseMetric, Metric


class BaseLoss(BaseMetric, abc.ABC):
    """
    The base loss function

    * extends: `..metrics.Metric`
    * implements: `..callbacks.protocols.Weighted`
    * abstract methods: `forward`
    * Loss tensor is stayed in memory until reset is called

    - Properties:
        - weight: A `float` of coeffiency applied to current loss function
    """

    __weight: float

    @property
    def weight(self) -> float:
        return self.__weight

    @weight.setter
    def weight(self, w: float) -> None:
        assert w >= 0, f"Weight must be a non-negative number, got {w}."
        self.__weight = w

    def __init__(self, target: str | None = None, weight: float = 1) -> None:
        """
        Constructor

        - Parameters:
            - target: An optional `str` of target name in `input` and `target` during direct calling
            - weight: A `float` of the loss weight
        """
        super().__init__(target=target)
        self.weight = weight

    def __call__(self, input: Any, target: Any) -> torch.Tensor:
        m: torch.Tensor = super().__call__(input, target) * self.weight
        self._results[-1] *= self.weight
        assert m.numel() == 1, _raise(TypeError(f"The returned loss must be a scalar tensor, got shape {m.shape}"))
        return m

    @abc.abstractmethod
    def forward(self, input: Any, target: Any) -> torch.Tensor:
        ...


class Loss(BaseLoss, Metric):
    """
    The main loss function

    * extends: `BaseLoss`, `..metrics.Metric`
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
    """

    def __init__(self, loss_fn: Callable[[Any, Any], torch.Tensor] | None = None, target: str | None = None, weight: float = 1) -> None:
        """
        Constructor

        - Parameters:
            - loss_fn: An optional `Callable` function that accepts input or `y_pred` in `Any` kind and target or `y_true` in `Any` kind as inputs and gives a loss in `torch.Tensor`
            - target: An optional `str` of target name in `input` and `target` during direct calling
            - weight: A `float` of the loss weight
        """
        super().__init__(target=target, weight=weight)
        self._metric_fn = loss_fn

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        return Metric.forward(self, input, target)


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

    def __init__(self, losses: list[Loss], target: str | None = None, weight: float = 1) -> None:
        """
        Constructor

        - Parameters:
            - losses: A `list` of `Loss` function
            - target: An optional `str` of target name in `input` and `target` during direct calling
            - weight: A `float` of the loss weight
        """
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

    def reset(self) -> None:
        for fn in self.losses:
            assert isinstance(fn, Loss), _raise(TypeError(f"Function {fn} is not a Loss object."))
            fn.reset()
        return super().reset()


L = TypeVar("L", bound=Loss)
P = TypeVar("P", bound=torch.nn.DataParallel)


class _RecordingDataParallel(torch.nn.DataParallel):
    """DataParallel that keeps replicas for downstream metric aggregation."""

    _replicas: list[torch.nn.Module] | None

    def __init__(self, module: torch.nn.Module, device_ids: list[int] | None = None, output_device: torch.device | None = None, dim: int = 0) -> None:
        super().__init__(module, device_ids=device_ids, output_device=output_device, dim=dim)
        self._replicas = None

    def forward(self, *inputs, **kwargs) -> Any:
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(f"module must have its parameters and buffers on device {self.src_device_obj} (device_ids[0]) but found one of them on device: {t.device}")

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            self._replicas = None
            return self.module(*inputs[0], **kwargs[0])

        replicas = self.replicate(self.module, self.device_ids[: len(inputs)])
        self._replicas = replicas
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)


class ParallelLoss(Loss, Generic[L, P]):
    """
    A data parallel loss function

    * extends: `torch.nn.parallel.DataParallel`
    * implements: `torchmanager_core.protocols.Resulting`

    - Properties:
        - module: A `torch.nn.Module` of the loss function
    """
    _metric_fn: P
    module: L

    def __init__(self, module: L, device_ids: list[int] | None = None, output_device: torch.device | None = None, dim: int = 0, *, parallel_type: type[P] = _RecordingDataParallel) -> None:
        super().__init__(parallel_type(module, device_ids, output_device, dim=dim))
        self.module = module

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        input = devices.move_to_device(input, devices.CPU)
        target = devices.move_to_device(target, devices.CPU)
        loss: torch.Tensor = super().forward(input, target)

        # merge individual loss breakdowns when running multi-loss in data parallel mode
        if isinstance(self.module, MultiLosses):
            self._merge_replica_losses()
        return loss.mean()

    def _merge_replica_losses(self) -> None:
        """Gather per-loss results from replicas created inside DataParallel forward."""
        # check if in data parallel
        if not isinstance(self._metric_fn, _RecordingDataParallel):
            return

        # get replicas
        replicas = self._metric_fn._replicas
        if not replicas:
            return

        # loop for replicas
        for replica in replicas:
            if not isinstance(replica, MultiLosses) or not isinstance(self.module, MultiLosses):
                continue
            for merged_loss, replica_loss in zip(self.module.losses, replica.losses):
                self._merge_loss_result(cast(Loss, merged_loss), cast(Loss, replica_loss))

        # clear replicas
        self._metric_fn._replicas = None

    @staticmethod
    def _merge_loss_result(target_loss: Loss, replica_loss: Loss) -> None:
        """Accumulate replica loss results into the master loss without re-computation."""
        target_loss.record(replica_loss.result)

    def reset(self) -> None:
        """Reset the current results list"""
        self.module.reset()
        super().reset()

    def to(self, device: torch.device) -> "ParallelLoss":
        """
        Move the loss to a specific device

        - Parameters:
            - device: A `torch.device` to move the loss to
        """
        self.module = self.module.to(device)
        return self


class _WrappedLoss(Loss):
    @property
    def wrapped_metric_fn(self) -> Callable[[Any, Any], torch.Tensor]:
        assert self._metric_fn is not None, _raise(AttributeError("Metric function is not given."))
        return self._metric_fn

    def __init__(self, loss_fn: Callable[[Any, Any], torch.Tensor], target: str | None = None, weight: float = 1) -> None:
        super().__init__(loss_fn, target, weight)

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        return self.wrapped_metric_fn(input, target)


def loss(fn: Callable[[Any, Any], torch.Tensor]) -> _WrappedLoss:
    """
    The loss wrapping function that wrap a function into a loss

    Use as a decorator:
    >>> import torch
    >>> @loss
    >>> def some_loss_fn(input: Any, target: Any) -> torch.Tensor:
    ...    return ...
    >>> manager = (..., loss_fn=some_loss_fn)
    """
    return _WrappedLoss(fn)


def loss_fn(target: str | None = None, weight: float = 1) -> Callable[[Callable[[Any, Any], torch.Tensor]], _WrappedLoss]:
    """
    The loss wrapping function that wrap a function with target and weight given into a loss

    Use as a decorator:
    >>> import torch
    >>> @loss_fn(target='out', weight=0.5)
    >>> def some_loss_fn(input: Any, target: Any) -> torch.Tensor:
    ...    return ...
    >>> manager = (..., loss_fn=some_loss_fn)
    """

    def wrapped_loss_fn(fn_to_wrap: Callable[[Any, Any], torch.Tensor]) -> _WrappedLoss:
        return _WrappedLoss(fn_to_wrap, target=target, weight=weight)

    return wrapped_loss_fn
