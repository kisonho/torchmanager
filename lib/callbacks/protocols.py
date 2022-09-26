from torchmanager_core import torch
from torchmanager_core.typing import Any, Enum, Optional, Protocol, runtime_checkable

from ..train.protocols import StateDictLoadable

@runtime_checkable
class ModelContainer(Protocol):
    """
    A container protocol that contains a property `model` as `torch.nn.module`
    """
    model: torch.nn.Module

class Frequency(Enum):
    """The frequency enum for callbacks"""
    EPOCH = 0
    BATCH = 1

class SummaryWriteble(Protocol):
    """The SummaryWriter protocol"""
    def add_scalars(self, main_tag: str, tag_scalar_dict: Any, global_step: Optional[int] = None) -> None: pass

class Weighted(Protocol):
    """A weigthted protocol that contains `weight` as its property"""
    weight: Any