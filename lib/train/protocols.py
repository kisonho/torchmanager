from torchmanager_core import abc
from torchmanager_core.typing import Any, List, OrderedDict, Protocol, Optional
from torchmanager_core.view.protocols import VerboseControllable

class LrSteping(Protocol):
    @abc.abstractmethod
    def get_last_lr(self) -> List[float]: return NotImplemented

    @abc.abstractmethod
    def step(self, epoch: Optional[int] = None) -> None: pass

class StateDictLoadable(Protocol):
    """An object that can load state dict"""
    @abc.abstractmethod
    def load_state_dict(self, state_dict: OrderedDict[str, Any], strict: bool = ...) -> Any: raise NotImplementedError

    @abc.abstractmethod
    def state_dict(self, prefix: str = ..., keep_vars: bool = ...) -> OrderedDict[str, Any]: return NotImplemented