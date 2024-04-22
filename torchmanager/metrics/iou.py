from torchmanager_core import torch, _raise
from torchmanager_core.typing import Optional

from .conf_mat import ConfusionMatrix
from .metric import Metric


class InstanceIoU(ConfusionMatrix):
    """
    The iIoU metric for segmentation

    * extends: `.conf_met.ConfusionMetrics`
    """

    def __init__(self, num_classes: int, /, *, target: Optional[str] = None) -> None:
        super().__init__(num_classes, target=target)

    @torch.no_grad()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # argmax for input
        input = super().forward(input, target)
        iou = torch.diag(input) / (input.sum(1) + input.sum(0) - torch.diag(input))
        return iou.nanmean()


class MeanIoU(Metric):
    """
    The mIoU metric for segmentation

    * extends: `torch.nn.Module`
    """
    _dim: int
    _smooth: float
    _threshold: float

    def __init__(self, dim: int = 1, smooth: float = 1e-4, threshold: float = 0, *, target: Optional[str] = None) -> None:
        """
        Constructor

        - Parameters:
            - dim: An `int` of class dimension
            - smooth: A `float` of smooth value to avoid zero devision
            - threshold: A `float` of min mIoU threshold
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(target=target)
        assert dim > 0, _raise(ValueError(f"The dimension must be a positive number, got {dim}."))
        assert threshold >= 0 and threshold <= 1, _raise(ValueError(f"The threshold must be in range [0,1], got {threshold}."))
        self._dim = dim
        self._smooth = smooth
        self._threshold = threshold

    @torch.no_grad()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate predictions from logits
        num_classes = input.shape[self._dim]

        if num_classes > 1:
            target_shape = input.shape
            input = input.argmax(self._dim)

            # initialize
            pred_masks = torch.zeros(*target_shape, device=input.device, dtype=torch.bool)
            true_masks = torch.zeros(*target_shape, device=target.device, dtype=torch.bool)
            
            # Convert indices to one-hot encoded masks
            for i in range(num_classes):
                pred_masks[:, i, ...] = (input == i)
                true_masks[:, i, ...] = (target == i)
        else:
            input = input > 0
            pred_masks = input
            true_masks = target

        # calculate iou
        intersection = (pred_masks & true_masks).float().sum()
        union = (pred_masks | true_masks).float().sum()
        iou = (intersection + self._smooth) / (union + self._smooth)
        thresholded = torch.clamp(10 / (1 - self._threshold) * (iou - self._threshold), 0, 10).ceil() / 10 if self._threshold > 0 else iou
        return thresholded
