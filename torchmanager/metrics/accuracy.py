from torchmanager_core import torch, Version
from torchmanager_core.protocols import Reduction
from torchmanager_core.typing import Optional

from .conf_met import BinaryConfusionMetric
from .metric import Metric


class Accuracy(Metric):
    """
    The traditional accuracy metric to compare two `torch.Tensor`

    * extends: `.metric.Metric`
    * implements: `torchmanager_core.protocols.VersionConvertible`
    """
    reduction: Reduction

    def __init__(self, *, reduction: Reduction = Reduction.MEAN, target: Optional[str] = None) -> None:
        super().__init__(target=target)
        self.reduction = reduction

    def convert(self, from_version: Version) -> None:
        if from_version < Version("v1.1"):
            self.reduction = Reduction.MEAN

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.reduction == Reduction.MEAN:
            return input.eq(target).to(torch.float32).mean()
        elif self.reduction == Reduction.SUM:
            return input.eq(target).to(torch.float32).sum()
        else:
            return input.eq(target)


class SparseCategoricalAccuracy(Accuracy):
    """
    The accuracy metric for normal integer labels

    * extends: `Accuracy`

    - Properties:
        - dim: An `int` of the probability dim index for the input
    """

    _dim: int

    def __init__(self, dim: int = -1, *, target: Optional[str] = None) -> None:
        """
        Constructor

        - Parameters:
            - dim: An `int` of the classification dimension
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(target=target)
        self._dim = dim

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input.argmax(dim=self._dim)
        return super().forward(input, target)


class CategoricalAccuracy(SparseCategoricalAccuracy):
    """
    The accuracy metric for categorical labels

    * extends: `SparseCategoricalAccuracy`
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.argmax(dim=self._dim)
        return super().forward(input, target)


class F1(BinaryConfusionMetric):
    """The F1 metrics"""

    def forward(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        # calculate precision and recall
        precision = tp / (tp + fp + self._eps)
        recall = tp / (tp + fn + self._eps)

        # calculate F1
        f1 = 2 * precision * recall / (precision + recall + self._eps)
        f1 = torch.mean(f1)
        return f1


class MAE(Metric):
    """
    The Mean Absolute Error metric

    * extends: `.metrics.Metric`

    - Properties:
        - reduction: A `.loss.Reduction` of reduction method
    """
    reduction: Reduction

    def __init__(self, *, reduction: Reduction = Reduction.MEAN, target: Optional[str] = None) -> None:
        """
        Constructor

        - Parameters:
            - reduction: A `.loss.Reduction` of reduction method
            - target: An optional `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(target=target)
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate MAE
        error = input - target
        error = error.abs()

        # error reduction
        if self.reduction == Reduction.MEAN:
            return error.mean()
        elif self.reduction == Reduction.SUM:
            return error.sum()
        else:
            return error


class Precision(BinaryConfusionMetric):
    """The Precision metrics"""

    def forward_metric(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        return tp / (tp + fp + self._eps)


class Recall(BinaryConfusionMetric):
    """The Recall metrics"""

    def forward_metric(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        return tp / (tp + fn + self._eps)
