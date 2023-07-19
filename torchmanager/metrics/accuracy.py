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


class Dice(BinaryConfusionMetric):
    """The dice score metrics"""
    def forward_metric(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        dice = 2 * tp / (2 * tp + fp + fn + self._eps)
        return dice.mean()


class F1(BinaryConfusionMetric):
    """The F1 metrics"""

    def forward_metric(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        # calculate precision and recall
        precision = tp / (tp + fp + self._eps)
        recall = tp / (tp + fn + self._eps)

        # calculate F1
        f1 = 2 * precision * recall / (precision + recall + self._eps)
        return f1.mean()


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


class PartialDice(Dice):
    """
    The partial dice score for a specific class index

    - Properties:
        - class_idx: An `int` of the target class index
    """
    class_idx: int

    def __init__(self, c: int, /, dim: int = 1, *, eps: float = 1e-7, target:Optional[str] = None):
        """
        Constructor

        - Parameters:
            - c: The target class index in `int`
            - dim: The class channel dimmension index in `int`
            - eps: A `float` of the small number to avoid zero divide
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(dim, eps=eps, target=target)
        self.class_idx = c

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input.argmax(self._dim)
        input_mask = input == self.class_idx
        target_mask = target == self.class_idx
        intersection = input_mask * target_mask
        dice = (2 * intersection.sum() + self._eps) / (input_mask.sum() + target_mask.sum() + self._eps)
        return dice.mean()


class Precision(BinaryConfusionMetric):
    """The Precision metrics"""

    def forward_metric(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        precision = tp / (tp + fp + self._eps)
        return precision.mean()


class Recall(BinaryConfusionMetric):
    """The Recall metrics"""

    def forward_metric(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        recall = tp / (tp + fn + self._eps)
        return recall.mean()
