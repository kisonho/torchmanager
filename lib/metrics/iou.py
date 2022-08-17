import warnings
import torch
from torchmanager_core.typing import Optional

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
    * [Pending Deprecation Warning]: The old `MIoU` metric in v1.0.3 calculates iIoU and has been renamed to `InstanceIoU` in v1.1.0, and will be removed in v1.2.0.
    """
    _dim: int
    _smooth: float

    def __init__(self, dim: int = 1, smooth: float = 1e-4, target: Optional[str] = None) -> None:
        """
        Constructor

        - Parameters:
            - dim: An `int` of class dimension
            - smooth: A `float` of smooth value to avoid zero devision
        """
        super().__init__(target=target)
        self._dim = dim
        self._smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input.argmax(self._dim)
        intersection = (input & target).float().sum()
        union = (input | target).float().sum()
        iou = (intersection + self._smooth) / (union + self._smooth)
        thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
        return thresholded

class MIoU(InstanceIoU):
    def __init__(self, num_classes: int, target: Optional[str] = None) -> None:
        super().__init__(num_classes, target)
        warnings.warn("The class `MIoU` will be renamed to `InstanceIoU` in v1.1.0, and will be removed in v1.2.0.", PendingDeprecationWarning)