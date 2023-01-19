from torchmanager_core import torch, deprecated
from torchmanager_core.typing import Optional

from .metric import Metric


class Accuracy(Metric):
    """
    The traditional accuracy metric to compare two `torch.Tensor`

    * extends: `.metric.Metric`
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return input.eq(target).to(torch.float32).mean()


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
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = input - target
        error = error.abs()
        return error.mean()
