from torchmanager_core import abc
from torchmanager_core.typing import Any, OrderedDict, Protocol, runtime_checkable

@runtime_checkable
class StateDictLoadable(Protocol):
    """An object that can load state dict"""
    @abc.abstractmethod
    def load_state_dict(self, state_dict: OrderedDict[str, Any]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def state_dict() -> OrderedDict[str, Any]:
        return NotImplemented