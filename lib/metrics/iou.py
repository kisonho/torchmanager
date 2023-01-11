from torchmanager_core import torch, _raise, deprecated
from torchmanager_core.typing import Optional
from torchmanager_core.view import warnings

from .conf_met import ConfusionMetrics
from .metric import Metric


class InstanceIoU(ConfusionMetrics):
    """
    The iIoU metric for segmentation

    * extends: `.conf_met.ConfusionMetrics`
    """

    def __init__(self, num_classes: int, target: Optional[str] = None) -> None:
        super().__init__(num_classes, target=target)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # argmax for input
        input = input.argmax(1).to(target.dtype)

        # calculate mean IoU
        hist = super().forward(input, target)
        iou = torch.diag(hist) / (hist.sum(1) + hist.sum(0) - torch.diag(hist))
        return iou.nanmean()


class MeanIoU(Metric):
    """
    The mIoU metric for segmentation

    * extends: `torch.nn.Module`
    * [Deprecation Warning]: The old `MIoU` metric in v1.0.3 calculates iIoU has been renamed to `InstanceIoU` in v1.1.0, and will be removed in v1.2.0.
    """
    _dim: int
    _smooth: float
    _threshold: float

    def __init__(self, dim: int = 1, smooth: float = 1e-4, threshold: float = 0, target: Optional[str] = None) -> None:
        """
        Constructor

        - Parameters:
            - dim: An `int` of class dimension
            - smooth: A `float` of smooth value to avoid zero devision
        """
        super().__init__(target=target)
        assert dim > 0, _raise(ValueError(f"The dimension must be a positive number, got {dim}."))
        assert threshold >= 0 and threshold <= 1, _raise(ValueError(f"The threshold must be in range [0,1], got {threshold}."))
        self._dim = dim
        self._smooth = smooth
        self._threshold = threshold

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input.argmax(self._dim)
        intersection = (input & target).float().sum()
        union = (input | target).float().sum()
        iou = (intersection + self._smooth) / (union + self._smooth)
        thresholded = torch.clamp(10 / (1 - self._threshold) * (iou - self._threshold), 0, 10).ceil() / 10
        return thresholded


@deprecated("v1.1.0", "v1.2.0")
class MIoU(InstanceIoU):
    def __init__(self, num_classes: int, target: Optional[str] = None) -> None:
        super().__init__(num_classes, target)
        warnings.warn("The class `MIoU` has been renamed to `InstanceIoU` in v1.1.0, and will be removed in v1.2.0.", DeprecationWarning)
