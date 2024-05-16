import abc, torch
from typing import Any, List, Optional, Protocol, runtime_checkable

@runtime_checkable
class DeviceMovable(Protocol):
    """The device movable protocol"""
    @abc.abstractmethod
    def to(self, device: torch.device) -> Any:
        raise NotImplementedError


class DataParallelType(Protocol):
    """The data paralleled protocol"""
    @abc.abstractmethod
    def __init__(self, module: Any, device_ids: Optional[List[int]] = None, output_device: Optional[torch.device] = None) -> None:
        raise NotImplementedError
