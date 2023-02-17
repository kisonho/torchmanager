from torchmanager_core import torch, deprecated
from torchmanager_core.typing import Optional

from .metric import Metric, Reduction


class Accuracy(Metric):
    """
    The traditional accuracy metric to compare two `torch.Tensor`

    * extends: `.metric.Metric`
    """
    reduction: Reduction

    def __init__(self, reduction: Reduction = Reduction.MEAN, target: Optional[str] = None) -> None:
        super().__init__(target=target)
        self.reduction = reduction

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
    * [Deprecation Warning]: The property `dim` has been deprecated from v1.1.0, and no longer be available in v1.2.0

    - Properties:
        - dim: An `int` of the probability dim index for the input
    """

    _dim: int

    @property
    @deprecated("v1.1.0", "v1.2.0")
    def dim(self) -> int:
        return self._dim

    @dim.setter
    @deprecated("v1.1.0", "v1.2.0")
    def dim(self, dim: int) -> None:
        self._dim = dim

    def __init__(self, dim: int = -1, target: Optional[str] = None) -> None:
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


class MAE(Metric):
    """
    The Mean Absolute Error metric

    * extends: `.metrics.Metric`

    - Properties:
        - reduction: A `.loss.Reduction` of reduction method
    """
    reduction: Reduction

    def __init__(self, reduction: Reduction = Reduction.MEAN, target: Optional[str] = None) -> None:
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
