from ..core import torch
from .metrics import Metric

class Accuracy(Metric):
    """The traditional accuracy metric to compare two `torch.Tensor`"""
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return input.eq(target).to(torch.float32).mean()

class SparseCategoricalAccuracy(Accuracy):
    """
    The accuracy metric for normal integer labels
    
    - Properties:
        - dim: An `int` of the probability dim index for the input
    """
    dim: int

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        calculate the accuracy
        
        - Parameters:
            - input: The prediction, or `y_pred`, in `Any` kind
            - target: The label, or `y_true`, in `Any` kind
        - Returns: The metric in `torch.Tensor`
        """
        input = input.argmax(dim=self.dim)
        return super().forward(input, target)

class CategoricalAccuracy(SparseCategoricalAccuracy):
    """The accuracy metric for categorical labels"""
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        calculate the accuracy
        
        - Parameters:
            - input: The prediction, or `y_pred`, in `Any` kind
            - target: The onehot label, or `y_true`, in `Any` kind
        - Returns: The metric in `torch.Tensor`
        """
        target = target.argmax(dim=self.dim)
        return super().forward(input, target)

class MAE(Metric):
    """The Mean Absolute Error metric"""
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = input - target
        error = error.abs()
        return error.mean()