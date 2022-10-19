from torchmanager_core import abc
from torchmanager_core.typing import Any, Enum, List, OrderedDict, Protocol, Optional
from torchmanager_core.view.protocols import VerboseControllable

class Frequency(Enum):
    """
    The frequency enum for learning rate
    
    * [Pending Deprecate Warning]: This enum will be deprecated from v1.1.0 and will be removed in v1.2.0.
    """
    EPOCH = 0
    BATCH = 1

LrScheduleFreq = Frequency

class LrSteping(Protocol):
    @abc.abstractmethod
    def get_last_lr(self) -> List[float]: return NotImplemented

    @abc.abstractmethod
    def step(self, epoch: Optional[int] = None) -> None: pass

class StateDictLoadable(Protocol):
    """An object that can load state dict"""
    @abc.abstractmethod
    def load_state_dict(self, state_dict: OrderedDict[str, Any], strict: bool = True) -> Any: return NotImplemented

    @abc.abstractmethod
    def state_dict(self, prefix: str = '', keep_vars: bool = False) -> OrderedDict[str, Any]: return NotImplemented