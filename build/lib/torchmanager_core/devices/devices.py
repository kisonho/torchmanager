from operator import index
from typing import Any, Iterable, Optional, Tuple, Union

import torch, warnings

from .protocols import DeviceMovable

CPU = torch.device('cpu')
'''The main CPU'''
DEFAULT = torch.cuda.current_device() if torch.cuda.is_available() else CPU
'''The default device'''
GPU = torch.device('cuda')
'''The overall CUDA devices'''
GPUS = [torch.device(i) for i in range(torch.cuda.device_count())]
'''The list of available CUDA devices'''

def data_parallel(raw_model: torch.nn.Module, *args, **kwargs) -> Tuple[Union[torch.nn.Module, torch.nn.parallel.DataParallel], bool]:
    """
    Make a `torch.nn.Module` data parallel

    - Parameters:
        - raw_model: A target `torch.nn.Module`
    - Returns: A `tuple` of either data paralleled `torch.nn.parallel.DataParallel` model if CUDA is available or a raw model if not, and a `bool` flag of if the model data paralleled successfuly.
    """
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(raw_model, *args, **kwargs)
        return model, True
    else:
        warnings.warn(f"[Device Warning]: CUDA is not available, unable to use multi-GPUs.", ResourceWarning)
        return raw_model, False

def empty_cache() -> None:
    """Empty CUDA cache"""
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def find(specified: Optional[torch.device] = None) -> Tuple[torch.device, torch.device]:
    """
    Find available devices

    - Pameters:
        - specified: An optional `torch.device` of specified
    - Returns: A `tuple` of CPU in `torch.device` and available device in `torch.device`
    """
    if specified is None:
        return (CPU, GPU) if torch.cuda.is_available() else (CPU, CPU)
    else:
        warnings.warn(f"[Device Warning]: Using specified device {specified}.", ResourceWarning)
        return CPU, specified

def move_to_device(target: Any, device: torch.device) -> Any:
    """
    Recurrently move a target variable to device if elements perform to `DeviceMovable` protocol
    
    - Parameters:
        - target: `Any` type of target
        - device: A `torch.device` of target device
    - Returns: The same type of target but moved to target device
    """
    if isinstance(target, DeviceMovable):
        target = target.to(device)
    elif isinstance(target, dict):
        target = {k: move_to_device(t, device) for k, t in target.items()}
    elif isinstance(target, Iterable):
        target = [move_to_device(t, device) for t in target]
    return target

def set_default(d: torch.device) -> None:
    if d.index is not None and d.type == "cuda": torch.cuda.set_device(d)