from torchmanager_core import torch
from torchmanager_core.typing import Any, Callable, Optional

from .loss import Loss, Reduction


class _ReductableLoss(Loss):
    """
    The MSE loss

    - Properties:
        - reduction: A `.loss.Reduction` of reduction method
        - replace_nan: A `boolean` flag of if replacing nan results to zeros
    """
    reduction: Reduction
    replace_nan: bool

    def __init__(self, loss_fn: Optional[Callable[[Any, Any], torch.Tensor]] = None, *, reduction: Reduction = Reduction.MEAN, replace_nan: bool = False, target: Optional[str] = None, weight: float = 1) -> None:
        """
        Constructor

        - Parameters:
            - reduction: A `.loss.Reduction` of reduction method
            - replace_nan: A `boolean` flag of if replacing nan results to zeros
            - target: An optional `str` of target name in `input` and `target` during direct calling
            - weight: A `float` of the loss weight
        """
        super().__init__(loss_fn, target=target, weight=weight)
        self.reduction = reduction
        self.replace_nan = replace_nan

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate mse loss
        loss = super().forward(input, target)

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


class MAE(_ReductableLoss):
    """
    The MSE loss
    """
    def __init__(self, *, reduction: Reduction = Reduction.MEAN, replace_nan: bool = False, target: Optional[str] = None, weight: float = 1) -> None:
        """
        Constructor

        - Parameters:
            - reduction: A `.loss.Reduction` of reduction method
            - replace_nan: A `boolean` flag of if replacing nan results to zeros
            - target: An optional `str` of target name in `input` and `target` during direct calling
            - weight: A `float` of the loss weight
        """
        l1 = torch.nn.L1Loss(reduction="none")
        super().__init__(l1, reduction=reduction, replace_nan=replace_nan, target=target, weight=weight)


class MSE(_ReductableLoss):
    """
    The MSE loss
    """
    def __init__(self, *, reduction: Reduction = Reduction.MEAN, replace_nan: bool = False, target: Optional[str] = None, weight: float = 1) -> None:
        """
        Constructor

        - Parameters:
            - reduction: A `.loss.Reduction` of reduction method
            - replace_nan: A `boolean` flag of if replacing nan results to zeros
            - target: An optional `str` of target name in `input` and `target` during direct calling
            - weight: A `float` of the loss weight
        """
        mse = torch.nn.MSELoss(reduction="none")
        super().__init__(mse, reduction=reduction, replace_nan=replace_nan, target=target, weight=weight)
