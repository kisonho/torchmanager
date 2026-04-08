import torch

from torchmanager_core.version import deprecated
from .hook import add_grad_clip, backward_hook
from .protocols import Removable

__all__ = ["add_grad_clip", "backward_hook"]

@deprecated("v1.5", "v2.0")
def add_gard_clip(model: torch.nn.Module, /, min_value: float, max_value: float, *, replace_nan: bool = False) -> list[Removable]:
    return add_grad_clip(model, min_value=min_value, max_value=max_value, replace_nan=replace_nan)
