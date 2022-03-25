# import typing modules
from __future__ import annotations
from typing import Any, Iterable, Optional, Protocol, Tuple, Union, runtime_checkable

# import required modules
import abc, torch, warnings

@runtime_checkable
class _DeviceMovable(Protocol):
    """The device movable protocol"""
    @abc.abstractmethod
    def to(self, device: torch.device) -> Any:
        raise NotImplementedError

def data_parallel(raw_model: torch.nn.Module) -> Tuple[Union[torch.nn.Module, torch.nn.parallel.DataParallel], bool]:
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(raw_model)
        return model, True
    else:
        warnings.warn(f"[Device Warning]: The use_multi_gpus flag is set to True, but CUDA is not available.", ResourceWarning)
        return raw_model, False

def empty_cache() -> None:
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def find(specified: Optional[torch.device] = None) -> Tuple[torch.device, torch.device]:
    """
    Find available devices

    - Pameters:
        - specified: An optional `torch.device` of specified
    - Returns: A `tuple` of CPU in `torch.device` and found device in `torch.device`
    """
    cpu = torch.device("cpu")
    if specified is None:
        gpu = torch.device("cuda")
        return (cpu, gpu) if torch.cuda.is_available() else (cpu, cpu)
    else:
        warnings.warn(f"[Device Warning]: Using specified device {specified}.", ResourceWarning)
        return cpu, specified

def move_to_device(target: Any, device: torch.device) -> Any:
    """
    Move a target variable to device
    
    - Parameters:
        - target: `Any` type of target
        - device: A `torch.device` of target device
    - Returns: The same type of target but moved to target device
    """
    if isinstance(target, _DeviceMovable):
        target = target.to(device)
    elif isinstance(target, dict):
        target = {k: move_to_device(t, device) for k, t in target.items()}
    elif isinstance(target, Iterable):
        target = [move_to_device(t, device) for t in target]
    return target