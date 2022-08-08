from typing import Any, Protocol, runtime_checkable

import abc, torch

@runtime_checkable
class DeviceMovable(Protocol):
    """The device movable protocol"""
    @abc.abstractmethod
    def to(self, device: torch.device) -> Any:
        raise NotImplementedError