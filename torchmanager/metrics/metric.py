from torchmanager_core import torch, API_VERSION, Version, _raise
from torchmanager_core.typing import Any, Callable, Generic, TypeVar


M = TypeVar("M", bound=Callable[[Any, Any], torch.Tensor] | None)


class Metric(torch.nn.Module, Generic[M]):
    """
    The basic metric class

    * extends: `torch.nn.Module`
    * implements: `torchmanager_core.protocols.Resulting`
    * Metric tensor is released from memory as soon as the result returned

    - Properties:
        - result: The `torch.Tensor` of average metric results
        - results: An optional `torch.Tensor` of all metric results
    """
    __result: torch.Tensor | float
    _metric_fn: M
    _results: list[torch.Tensor]
    _target: str | None

    @property
    def result(self) -> torch.Tensor:
        if len(self._results) > 0:
            return torch.tensor(self.__result / len(self._results))
        else:
            return torch.tensor(torch.nan)

    @property
    def results(self) -> torch.Tensor | None:
        if len(self._results) > 0:
            return torch.concat(self._results)
        else:
            return None

    def __init__(self, metric_fn: M = None, target: str | None = None) -> None:
        """
        Constructor

        - Parameters:
            - metric_fn: An optional `Callable` metrics function that accepts `Any` kind of prediction input and target and returns a metric `torch.Tensor`. A `call` method must be overriden if this parameter is set as `None`.
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        super().__init__()
        self.__result = 0
        self._metric_fn = metric_fn
        self._results = []
        self._target = target

    def __call__(self, input: Any, target: Any) -> torch.Tensor:
        # unpack input and target
        input = input[self._target] if self._target is not None and isinstance(input, dict) else input
        target = target[self._target] if self._target is not None and isinstance(target, dict) else target

        # call
        m: torch.Tensor = super().__call__(input, target)
        self._results.append(m.unsqueeze(0).cpu().detach())
        self.__result += self._results[-1]
        return m

    def convert(self, from_version: Version) -> None:
        if from_version < API_VERSION:
            self.__result = 0
        pass

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        """
        Forward the current result method

        - Parameters:
            - input: The prediction, or `y_pred`, in `Any` kind
            - target: The label, or `y_true`, in `Any` kind
        - Returns: The metric in `torch.Tensor`
        """
        # main method
        if self._metric_fn is not None:
            return self._metric_fn(input, target)
        else:
            raise NotImplementedError("metric_fn is not given.")

    def reset(self) -> None:
        """Reset the current results list"""
        self.__result = 0
        self._results.clear()


WRAPPED_M = TypeVar("WRAPPED_M", bound=Callable[[Any, Any], torch.Tensor])


class _WrappedMetric(Metric[WRAPPED_M]):
    @property
    def wrapped_metric_fn(self) -> WRAPPED_M:
        assert self._metric_fn is not None, _raise(AttributeError("Metric function is not given."))
        return self._metric_fn

    def __init__(self, metric_fn: WRAPPED_M, target: str | None = None) -> None:
        super().__init__(metric_fn, target)

    @torch.no_grad()
    def forward(self, input: Any, target: Any) -> torch.Tensor:
        return self.wrapped_metric_fn(input, target)


def metric(fn: Callable[[Any, Any], torch.Tensor]) -> _WrappedMetric:
    """
    The metric wrapping function that wrap a function into a metric

    Use as a decorator:
    >>> import torch
    >>> @metric
    >>> def some_metric_fn(input: Any, target: Any) -> torch.Tensor:
    ...    return ...
    >>> manager = (..., metric_fns={'out': some_metric_fn})
    """
    return _WrappedMetric(fn)


def metric_fn(target: str | None = None) -> Callable[[Callable[[Any, Any], torch.Tensor]], _WrappedMetric]:
    """
    The loss wrapping function that wrap a function with target and weight given into a loss

    Use as a decorator:
    >>> import torch
    >>> @metric_fn(target='out')
    >>> def some_metric_fn(input: Any, target: Any) -> torch.Tensor:
    ...    return ...
    >>> manager = (..., metric_fns={'out': some_metric_fn})
    """
    def wrapped_fn(fn_to_wrap: Callable[[Any, Any], torch.Tensor]) -> _WrappedMetric:
        return _WrappedMetric(fn_to_wrap, target=target)
    return wrapped_fn
