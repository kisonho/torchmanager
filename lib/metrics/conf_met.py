from ..core import torch
from .metrics import Metric

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