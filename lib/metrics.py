from .core import torch, view
from .core._typing import Any, Callable, List, Optional

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
    _results: List[torch.Tensor]

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
        self._results.append(m.detach())
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

class Accuracy(Metric):
    """The traditional accuracy metric to compare two `torch.Tensor`"""
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return input.eq(target).to(torch.float32).mean()

class ConfusionMetrics(Metric):
    """The mIoU metric for segmentation"""
    __num_classes: int

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.__num_classes = num_classes

    def _fast_hist(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate historgram
        
        - Parameters:
            - input: The prediction `torch.Tensor`, or `y_pred`
            - target: The label `torch.Tensor`, or `y_true`
        - Returns: A `torch.Tensor` of historgram
        """
        mask = (target >= 0) & (target < self.__num_classes)
        hist = torch.bincount(self.__num_classes * target[mask].to(torch.int64) + input[mask], minlength=self.__num_classes ** 2).reshape(self.__num_classes, self.__num_classes)
        return hist

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # initialize metrics
        conf_mat = torch.zeros((self.__num_classes, self.__num_classes), device=input.device)

        # add confusion metrics
        for y_pred, y_true in zip(input, target):
            y_pred: torch.Tensor
            y_true: torch.Tensor
            conf_mat += self._fast_hist(y_pred.flatten(), y_true.flatten())
        
        # calculate mean IoU
        return conf_mat

class MAE(Metric):
    """The Mean Absolute Error metric"""
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = input - target
        error = error.abs()
        return error.mean()

class MIoU(ConfusionMetrics):
    """The mIoU metric for segmentation"""
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # argmax for input
        input = input.argmax(1).to(target.dtype)
        
        # calculate mean IoU
        hist = super().forward(input, target)
        iou = torch.diag(hist) / (hist.sum(1) + hist.sum(0) - torch.diag(hist))
        return iou.nanmean()

class SparseCategoricalAccuracy(Accuracy):
    """
    The accuracy metric for normal integer labels
    
    - Properties:
        - dim: An `int` of the probability dim index for the input
    """
    dim: int

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        calculate the accuracy
        
        - Parameters:
            - input: The prediction, or `y_pred`, in `Any` kind
            - target: The label, or `y_true`, in `Any` kind
        - Returns: The metric in `torch.Tensor`
        """
        input = input.argmax(dim=self.dim)
        return super().forward(input, target)

class CategoricalAccuracy(SparseCategoricalAccuracy):
    """The accuracy metric for categorical labels"""
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        calculate the accuracy
        
        - Parameters:
            - input: The prediction, or `y_pred`, in `Any` kind
            - target: The onehot label, or `y_true`, in `Any` kind
        - Returns: The metric in `torch.Tensor`
        """
        target = target.argmax(dim=self.dim)
        return super().forward(input, target)

def metric(fn: Callable[[Any, Any], torch.Tensor]) -> Metric:
    """
    The metric wrapping function that wrap a function into a metric

    * Use as a decorator
    """
    return Metric(fn)