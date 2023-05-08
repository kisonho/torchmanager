import abc, torch, sys
from enum import Enum
from typing import Any, Dict, List, Optional, OrderedDict, Protocol, runtime_checkable
from typing_extensions import Self

from .devices.protocols import DeviceMovable
from .view.protocols import VerboseControllable


class Frequency(Enum):
    """
    The frequency enum for learning rate
    """

    EPOCH = 0
    BATCH = 1


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


class LrSteping(Protocol):
    @abc.abstractmethod
    def get_last_lr(self) -> List[float]:
        return NotImplemented

    @abc.abstractmethod
    def step(self, epoch: Optional[int] = None) -> None:
        pass


@runtime_checkable
class ModelContainer(Protocol):
    """A container protocol that contains a property `model` as `torch.nn.module`"""

    model: torch.nn.Module


class StateDictLoadable(Protocol):
    """An object that can load state dict"""

    @abc.abstractmethod
    def load_state_dict(self, state_dict: OrderedDict[str, Any], strict: bool = True) -> Any:
        return NotImplemented

    @abc.abstractmethod
    def state_dict(self, *, prefix: str = "", keep_vars: bool = False) -> Dict[str, Any]:
        return NotImplemented


class Trainable(Protocol):
    """An object that can switch training mode"""
    @abc.abstractmethod
    def eval(self) -> Self:
        return NotImplemented

    @abc.abstractmethod
    def train(self, mode: bool = True) -> Self:
        return NotImplemented

class Resulting(StateDictLoadable, Trainable, Protocol):
    """An object that have result available with reset method"""
    @property
    @abc.abstractmethod
    def _target(self) -> Optional[str]:
        return NotImplemented

    @_target.setter
    @abc.abstractmethod
    def _target(self, t: Optional[str]) -> None:
        pass

    @property
    @abc.abstractmethod
    def result(self) -> torch.Tensor:
        return NotImplemented

    @abc.abstractmethod
    def __call__(self, input: Any, target: Any) -> torch.Tensor:
        return NotImplemented

    @abc.abstractmethod
    def reset(self) -> None:
        pass


class SummaryWriteble(Protocol):
    """The SummaryWriter protocol"""

    @abc.abstractmethod
    def add_scalars(self, main_tag: str, tag_scalar_dict: Any, global_step: Optional[int] = None) -> None:
        pass


class Weighted(Protocol):
    """A weigthted protocol that contains `weight` as its property"""

    weight: Any
