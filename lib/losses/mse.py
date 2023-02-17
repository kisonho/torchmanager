from torchmanager_core import torch
from torchmanager_core.typing import Optional

from .loss import Loss, Reduction


class MSE(Loss):
    reduction: Reduction

    def __init__(self, reduction: Reduction = Reduction.MEAN, target: Optional[str] = None, weight: float = 1) -> None:
        mse = torch.nn.MSELoss(reduction="none")
        super().__init__(mse, target=target, weight=weight)
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate mse loss
        loss = (input - target) ** 2
        max_value = torch.finfo(loss.dtype).max
        loss = loss.nan_to_num(0, posinf=max_value, neginf=0)

        # reduction
        if self.reduction == Reduction.MEAN:
            return loss.mean()
        elif self.reduction == Reduction.SUM:
            return loss.sum()
        else:
            return loss
