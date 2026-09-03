import torch
from typing import Any, Optional, Protocol, runtime_checkable


class DataParallelType(Protocol):
    """The data paralleled protocol"""
    def __init__(self, module: Any, device_ids: Optional[list[int]] = None, output_device: Optional[torch.device] = None) -> None:
        ...


class DataParallelable(Protocol):
    """The data parallel delegate protocol"""
    def data_parallel(self, target_devices: list[torch.device], *, parallel_type: type[torch.nn.parallel.DataParallel] = torch.nn.parallel.DataParallel) -> bool:
        ...


@runtime_checkable
class DeviceMovable(Protocol):
    """The device movable protocol"""
    def to(self, device: torch.device) -> Any:
        ...

