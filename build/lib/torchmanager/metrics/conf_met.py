from torchmanager_core import torch, _raise
from torchmanager_core.typing import Optional

from .metric import Metric


class Histogram(Metric):
    """
    The metric that calculates histogram

    * extends: `.metric.Metric`

    - Properties:
        - num_classes: An `int` of the total number of classes
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

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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


class ConfusionMetrics(Histogram):
    """
    The metric that calculates confusion metrics

    * extends: `Histogram`
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # initialize metrics
        conf_mat = torch.zeros((self.num_classes, self.num_classes), device=input.device)

        # add confusion metrics
        for y_pred, y_true in zip(input, target):
            y_pred: torch.Tensor
            y_true: torch.Tensor
            conf_mat += super().forward(y_pred.flatten(), y_true.flatten())

        # calculate mean IoU
        return conf_mat
