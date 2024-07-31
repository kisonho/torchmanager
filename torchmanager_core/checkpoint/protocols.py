import abc, torch
from typing import Any, Callable, Optional, OrderedDict, Protocol, runtime_checkable


class StateDictLoadable(Protocol):
    """An object that can load state dict"""

    @abc.abstractmethod
    def load_state_dict(self, *, state_dict: OrderedDict[str, Any], strict: bool = True) -> Any:
        ...

    @abc.abstractmethod
    def state_dict(self, *, prefix: str = "", keep_vars: bool = False) -> dict[str, Any]:
        ...


@runtime_checkable
class ModelContainer(Protocol):
    """A container protocol that contains a property `model` as `torch.nn.module`"""

    model: torch.nn.Module
    optimizer: Optional[torch.optim.Optimizer]
    loss_fn: Optional[StateDictLoadable]
    metric_fns: dict[str, StateDictLoadable]


@runtime_checkable
class WrappedFn(Protocol):
    @property
    @abc.abstractmethod
    def wrapped_metric_fn(self) -> Callable[[Any, Any], torch.Tensor]:
        ...
