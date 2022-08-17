from torchmanager_core import abc
from torchmanager_core.typing import Any, Enum, Optional, Protocol

class Frequency(Enum):
    """The frequency enum for callbacks"""
    EPOCH = 0
    BATCH = 1

class SummaryWriteble(Protocol):
    """The SummaryWriter protocol"""
    def add_scalars(self, main_tag: str, tag_scalar_dict: Any, global_step: Optional[int] = None) -> None: pass

class Weighted(Protocol):
    @property
    @abc.abstractmethod
    def weight(self) -> Any: return NotImplemented

    @weight.setter
    @abc.abstractmethod
    def weight(self, w: Any) -> None: pass