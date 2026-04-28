from torchmanager_core import torch
from torchmanager_core.protocols import Reduction
from torchmanager_core.typing import Any, Callable, TypeVar

from .loss import Loss

__all__ = ["Identity", "MAE", "MSE"]

LossFn = TypeVar("LossFn", bound=Callable[[Any, Any], torch.Tensor] | None)


class _ReductableLoss(Loss[LossFn]):
    """
    The loss that reduct its dimension with a specific method.

    - Properties:
        - reduction: A `.loss.Reduction` of reduction method
        - replace_nan: A `boolean` flag of if replacing nan results to zeros
    """
    reduction: Reduction
    replace_nan: bool

    def __init__(self, loss_fn: LossFn = None, *, reduction: Reduction = Reduction.MEAN, replace_nan: bool = False, target: str | None = None, weight: float = 1) -> None:
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
        match self.reduction:
            case Reduction.MEAN:
                return loss.mean()
            case Reduction.SUM:
                return loss.sum()
            case Reduction.NONE:
                return loss


class Identity(_ReductableLoss[None]):
    """ An identity loss function that returns the input as the output."""
    def __init__(self, loss_fn: None = None, *, target: str | None = None, weight: float = 1) -> None:
        super().__init__(loss_fn, reduction=Reduction.NONE, replace_nan=False, target=target, weight=weight)

    def forward(self, input: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return input


class MAE(_ReductableLoss[torch.nn.L1Loss]):
    """
    The MSE loss
    """
    def __init__(self, *, reduction: Reduction = Reduction.MEAN, replace_nan: bool = False, target: str | None = None, weight: float = 1) -> None:
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


class MSE(_ReductableLoss[torch.nn.MSELoss]):
    """
    The MSE loss
    """
    def __init__(self, *, reduction: Reduction = Reduction.MEAN, replace_nan: bool = False, target: str | None = None, weight: float = 1) -> None:
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
