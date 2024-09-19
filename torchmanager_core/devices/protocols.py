import torch
from typing import Any, Optional, Protocol, runtime_checkable

@runtime_checkable
class DeviceMovable(Protocol):
    """The device movable protocol"""
    def to(self, device: torch.device) -> Any:
        ...


class DataParallelType(Protocol):
    """The data paralleled protocol"""
    def __init__(self, module: Any, device_ids: Optional[list[int]] = None, output_device: Optional[torch.device] = None) -> None:
        ...
