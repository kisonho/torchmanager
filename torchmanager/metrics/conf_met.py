from torchmanager_core import abc, torch, _raise, deprecated
from torchmanager_core.typing import Any, Optional

from .metric import Metric


class ConfusionMetrics(Metric, abc.ABC):
    """
    The metric that forward confusion metrics calculated by given `input` and `target` as final `input` in `forward` method

    * Extends: `.metric.Metric`
    * Abstract class

    - Properties:
        - num_classes: An `int` of the total number of classes
    - Methods to implement:
        - forward: The main forward function to calculate final metric as `torch.Tensor`, which accepts the confusion metrics of `torch.Tensor` with the label of `torch.Tensor
    """
    __num_classes: int

    @property
    def num_classes(self) -> int:
        return self.__num_classes

    def __init__(self, num_classes: int, target: Optional[str] = None) -> None:
        """
        Constructor

        - Parameters:
            - num_classes: An `int` of the total number of classes
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(target=target)
        assert num_classes > 0, _raise(ValueError(f"The number of classes must be a positive number, got {num_classes}."))
        self.__num_classes = num_classes

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # initialize metrics
        conf_mat = torch.zeros((self.num_classes, self.num_classes), device=input.device)

        # add confusion metrics
        for y_pred, y_true in zip(input, target):
            y_pred = y_pred.argmax(-1)
            conf_mat += self.forward_hist(y_pred.flatten(), y_true.flatten())

        # calculate final metric
        return super().__call__(conf_mat, target)

    @abc.abstractmethod
    def forward(self, input: Any, target: Any) -> torch.Tensor:
        return NotImplemented

    def forward_hist(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate historgram

        - Parameters:
            - input: The prediction `torch.Tensor`, or `y_pred`
            - target: The label `torch.Tensor`, or `y_true`
        - Returns: A `torch.Tensor` of historgram
        """
        mask = (target >= 0) & (target < self.num_classes)
        hist = torch.bincount(self.num_classes * target[mask].to(torch.int64) + input[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist


@deprecated('v1.2', 'v1.3')
class Histogram(ConfusionMetrics):
    pass
