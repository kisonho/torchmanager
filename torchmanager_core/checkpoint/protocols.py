import torch
from torch.optim.optimizer import Optimizer
from typing import Any, Callable, Mapping, Optional, OrderedDict, Protocol, runtime_checkable, overload


class StateDictLoadable(Protocol):
    """An object that can load state dict"""

    @overload
    def load_state_dict(self, *, state_dict: OrderedDict[str, Any]) -> Any:
        ...

    @overload
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> Any:
        ...

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> Any:
        ...

    def state_dict(self, *, prefix: str = "", keep_vars: bool = False) -> dict[str, Any]:
        ...


@runtime_checkable
class ModelContainer(Protocol):
    """A container protocol that contains a property `model` as `torch.nn.module`"""

    model: torch.nn.Module
    optimizer: Optional[Optimizer]
    loss_fn: Optional[StateDictLoadable]
    metric_fns: dict[str, StateDictLoadable]


@runtime_checkable
class WrappedFn(Protocol):
    @property
    def wrapped_metric_fn(self) -> Callable[[Any, Any], torch.Tensor]:
        ...
