from ..core import torch, view
from ..core.typing import Any, Callable, List, Optional

class Metric(torch.nn.Module):
    """
    The basic metric class

    * Could be use as a decorator of a function
    * Metric tensor is released from memory as soon as the result returned
    * [Deprecation Warning]: Method `call` is deprecated from v1.0.0 and will be removed from v1.1.0, override the `forward` method instead."

    - Parameters:
        - result: The `torch.Tensor` of average metric results
    """
    # properties
    _metric_fn: Optional[Callable[[Any, Any], torch.Tensor]]
    _results: List[float]

    @property
    def result(self) -> torch.Tensor:
        return torch.tensor(self._results).mean()

    def __init__(self, metric_fn: Optional[Callable[[Any, Any], torch.Tensor]]=None) -> None:
        """
        Constructor

        - Parameters:
            - metric_fn: An optional `Callable` metrics function that accepts `Any` kind of prediction input and target and returns a metric `torch.Tensor`. A `call` method must be overriden if this parameter is set as `None`.
        """
        super().__init__()
        self._results = []
        self._metric_fn = metric_fn

    def __call__(self, input: Any, target: Any) -> torch.Tensor:
        m: torch.Tensor = super().__call__(input, target)
        self._results.append(float(m.detach()))
        return m

    def call(self, input: Any, target: Any) -> torch.Tensor:
        return NotImplemented

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        """
        Forward the current result method
        
        - Parameters:
            - input: The prediction, or `y_pred`, in `Any` kind
            - target: The label, or `y_true`, in `Any` kind
        - Returns: The metric in `torch.Tensor`
        """
        # run deprecated method
        m = self.call(input, target)
        if m != NotImplemented:
            view.warnings.warn("[Deprecated Warning]: Method `call` was deprecated since version v1.0.0, the public method will be removed from 1.1.0. Override `forward` instead.", DeprecationWarning)
            return m

        # main method
        if self._metric_fn is not None:
            return self._metric_fn(input, target)
        else: raise NotImplementedError("[Metric Error]: metric_fn is not given.")
    
    def reset(self) -> None:
        """Reset the current results list"""
        self._results.clear()

def metric(fn: Callable[[Any, Any], torch.Tensor]) -> Metric:
    """
    The metric wrapping function that wrap a function into a metric

    * Use as a decorator
    """
    return Metric(fn)