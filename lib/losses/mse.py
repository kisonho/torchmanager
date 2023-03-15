from torchmanager_core import torch
from torchmanager_core.typing import Optional

from .loss import Loss, Reduction


class MSE(Loss):
    """
    The MSE loss

    - Properties:
        - reduction: A `.loss.Reduction` of reduction method
        - replace_nan: A `boolean` flag of if replacing nan results to zeros
    """
    reduction: Reduction
    replace_nan: bool

    def __init__(self, reduction: Reduction = Reduction.MEAN, replace_nan: bool = False, target: Optional[str] = None, weight: float = 1) -> None:
        """
        Constructor

        - Parameters:
            - reduction: A `.loss.Reduction` of reduction method
            - replace_nan: A `boolean` flag of if replacing nan results to zeros
            - target: An optional `str` of target name in `input` and `target` during direct calling
            - weight: A `float` of the loss weight
        """
        mse = torch.nn.MSELoss(reduction="none")
        super().__init__(mse, target=target, weight=weight)
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate mse loss
        loss = (input - target) ** 2

        # check NAN
        if self.replace_nan:
            max_value = torch.finfo(loss.dtype).max
            loss = loss.nan_to_num(0, posinf=max_value, neginf=-1)

        # reduction
        if self.reduction == Reduction.MEAN:
            return loss.mean()
        elif self.reduction == Reduction.SUM:
            return loss.sum()
        else:
            return loss