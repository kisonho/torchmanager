from torchmanager_core import torch, _raise
from torchmanager_core.typing import Any, Callable, List, Optional


class Metric(torch.nn.Module):
    """
    The basic metric class

    * extends: `torch.nn.Module`
    * implements: `torchmanager_core.protocols.Resulting`
    * Metric tensor is released from memory as soon as the result returned

    - Properties:
        - result: The `torch.Tensor` of average metric results
        - results: An optional `torch.Tensor` of all metric results
    """
    _metric_fn: Optional[Callable[[Any, Any], torch.Tensor]]
    _results: List[torch.Tensor]
    _target: Optional[str]

    @property
    def result(self) -> torch.Tensor:
        if len(self._results) > 0:
            return torch.concat(self._results).mean()
        else:
            return torch.tensor(torch.nan)

    @property
    def results(self) -> Optional[torch.Tensor]:
        if len(self._results) > 0:
            return torch.concat(self._results)
        else:
            return None

    def __init__(self, metric_fn: Optional[Callable[[Any, Any], torch.Tensor]] = None, target: Optional[str] = None) -> None:
        """
        Constructor

        - Parameters:
            - metric_fn: An optional `Callable` metrics function that accepts `Any` kind of prediction input and target and returns a metric `torch.Tensor`. A `call` method must be overriden if this parameter is set as `None`.
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        super().__init__()
        self._metric_fn = metric_fn
        self._results = []
        self._target = target

    def __call__(self, input: Any, target: Any) -> torch.Tensor:
        # unpack input and target
        if self._target is not None:
            assert isinstance(input, dict) and isinstance(target, dict), _raise(TypeError(f"Given input or target must be dictionaries, got {type(input)} and {type(target)}."))
            assert self._target in input and self._target in target, _raise(TypeError(f"Target ‘{self._target}’ cannot be found not in input or target"))
            input, target = input[self._target], target[self._target]

        # call
        m: torch.Tensor = super().__call__(input, target)
        self._results.append(m.unsqueeze(0).detach())
        return m

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
        self._results.clear()


def metric(fn: Callable[[Any, Any], torch.Tensor]) -> Metric:
    """
    The metric wrapping function that wrap a function into a metric

    Use as a decorator:
    >>> import torch
    >>> @metric
    >>> def some_metric_fn(input: Any, target: Any) -> torch.Tensor:
    ...    return ...
    >>> manager = (..., metric_fns={'out': some_metric_fn})
    """
    return Metric(fn)


def metric_fn(target: Optional[str] = None) -> Callable[[Callable[[Any, Any], torch.Tensor]], Metric]:
    """
    The loss wrapping function that wrap a function with target and weight given into a loss

    Use as a decorator:
    >>> import torch
    >>> @metric_fn(target='out')
    >>> def some_metric_fn(input: Any, target: Any) -> torch.Tensor:
    ...    return ...
    >>> manager = (..., metric_fns={'out': some_metric_fn})
    """
    def wrapped_fn(fn_to_wrap: Callable[[Any, Any], torch.Tensor]) -> Metric:
        return Metric(fn_to_wrap, target=target)
    return wrapped_fn
