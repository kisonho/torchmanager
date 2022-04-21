from ..core import functional as F, torch
from ..core.typing import Any
from .losses import Loss

class CrossEntropy(Loss):
    """The cross entropy loss"""
    def __init__(self, *args, **kwargs) -> None:
        loss_fn = torch.nn.CrossEntropyLoss(*args, **kwargs)
        super().__init__(loss_fn)

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        assert self._metric_fn is not None, "[Loss Error]: Crossentropy loss has not been initialized."
        return self._metric_fn(input, target)

class FocalCrossEntropy(Loss):
    """The focal cross entropy loss"""
    _alpha: float
    _gamma: float
    _calculate_average: bool
    _ignore_index: int

    def __init__(self, alpha: float = 1, gamma: float = 0, calculate_average: bool = True, ignore_index: int = 255):
        """
        Constructor

        - Parameters:
            - alpha: A `float` of alpha in focal cross entropy
            - gamma: A `float` of gamma in focal cross entropy
            - calculate_average: A `bool` flag of if calculate average for the focal loss
        """
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_index = ignore_index
        self._calculate_average = calculate_average

    def forward(self, inputs: Any, targets: Any) -> torch.Tensor:
        # calculate loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self._ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss: torch.Tensor = self._alpha * (1 - pt) ** self._gamma * ce_loss

        # calculate average
        if self._calculate_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class KLDiv(Loss):
    """KL-Div Loss"""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        loss_fn = torch.nn.KLDivLoss(*args, **kwargs)
        super().__init__(loss_fn)