import torch, sys
from enum import Enum
from typing import Any, Callable, Optional, Protocol, runtime_checkable
from typing_extensions import Self

from .checkpoint import Checkpoint
from .checkpoint.protocols import ModelContainer, StateDictLoadable, WrappedFn
from .devices.protocols import DeviceMovable
from .version import Version
from .view.protocols import VerboseControllable


@runtime_checkable
class CkptConvertable(StateDictLoadable, Protocol):
    """A protocol for object that can be converted to checkpoint"""
    def to_checkpoint(self) -> Checkpoint[Self]:
        ...


class Frequency(Enum):
    """
    The frequency enum for learning rate
    """

    EPOCH = 0
    EPOCH_START = -1
    BATCH = 1
    BATCH_START = 2


class MonitorType(Enum):
    """The enum of monitor types"""

    MIN = int(0)
    MAX = int(1)

    @property
    def init_score(self) -> float:
        if self == MonitorType.MAX:
            return -1
        elif self == MonitorType.MIN:
            return sys.float_info.max
        else:
            raise TypeError(f"Monitor type {self} is not supported.")


class Steppable(Protocol):
    """An object that can step"""
    def step(self, *args, **kwargs) -> Any:
        ...


class LrSteping(Steppable):
    def get_last_lr(self) -> list[float]:
        ...

    def step(self, epoch: Optional[int] = None) -> None:
        ...


class Trainable(Protocol):
    """An object that can switch training mode"""
    def eval(self) -> Self:
        ...

    def train(self, mode: bool = True) -> Self:
        ...


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


class Resulting(DeviceMovable, StateDictLoadable, Trainable, Protocol):
    """An object that have result available with reset method"""
    _metric_fn: Optional[Callable[[Any, Any], torch.Tensor]]
    _target: Optional[str]

    @property
    def result(self) -> torch.Tensor:
        ...

    @property
    def results(self) -> Optional[torch.Tensor]:
        ...

    def __call__(self, input: Any, target: Any) -> torch.Tensor:
        ...

    def convert(self, from_version: Version) -> None:
        ...

    def reset(self) -> None:
        ...


class SummaryWriteble(Protocol):
    """The SummaryWriter protocol"""

    def add_scalars(self, main_tag: str, tag_scalar_dict: Any, global_step: Optional[int] = None) -> None:
        ...


class Weighted(Protocol):
    """A weigthted protocol that contains `weight` as its property"""

    weight: Any
