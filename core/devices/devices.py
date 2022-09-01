from typing import Any, Iterable, Optional, Tuple, TypeVar, Union

import torch, warnings

from .protocols import DeviceMovable

CPU = torch.device('cpu')
'''The main CPU'''
DEFAULT = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else CPU
'''The default device'''
GPU = torch.device('cuda')
'''The overall CUDA devices'''
GPUS = [torch.device(i) for i in range(torch.cuda.device_count())]
'''The list of available CUDA devices'''

def data_parallel(raw_model: torch.nn.Module, devices: list[torch.device] = GPUS, output_device: Optional[torch.device] = None) -> Tuple[Union[torch.nn.Module, torch.nn.parallel.DataParallel], bool]:
    """
    Make a `torch.nn.Module` data parallel

    - Parameters:
        - raw_model: A target `torch.nn.Module`
    - Returns: A `tuple` of either data paralleled `torch.nn.parallel.DataParallel` model if CUDA is available or a raw model if not, and a `bool` flag of if the model data paralleled successfuly.
    """
    if isinstance(raw_model, torch.nn.parallel.DataParallel):
        return raw_model, True
    elif torch.cuda.is_available():
        device_ids = [d.index for d in devices]
        model = torch.nn.parallel.DataParallel(raw_model, device_ids=device_ids, output_device=output_device)
        return model, True
    else:
        warnings.warn(f"CUDA is not available, unable to use multi-GPUs.", ResourceWarning)
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
    warnings.warn("This method will be deprecated from v1.1.0 and will be removed in v1.2.0, use `torchmanager_core.devices.search` instead.", PendingDeprecationWarning)
    if specified is None:
        return (CPU, GPU) if torch.cuda.is_available() else (CPU, CPU)
    else: return CPU, specified

def search(specified: Optional[Union[torch.device, list[torch.device]]] = None) -> Tuple[torch.device, torch.device, list[torch.device]]:
    """
    Find available devices

    - Pameters:
        - specified: An optional `torch.device` of specified
    - Returns: A `tuple` of CPU in `torch.device`, available device in `torch.device` and `list` of specified devices in `torch.device`
    """
    if specified is None:
        return (CPU, GPU, GPUS) if len(GPUS) > 0 else (CPU, CPU, [CPU])
    else:
        if not isinstance(specified, list):
            device = specified
            specified = [specified]
        else: device = GPU
        return CPU, device, specified

def move_to_device(target: Union[DeviceMovable, torch.optim.Optimizer,  dict[str, Any], list[Any]], device: torch.device) -> Union[DeviceMovable, torch.optim.Optimizer, dict[str, Any], list[Any]]:
    """
    Recurrently move a target variable to device if elements perform to `DeviceMovable` protocol
    
    - Parameters:
        - target: Either a target in `DeviceMovable`, a `torch.optim.Optimizer`, a `dict` of targets, or a `list` of targets, targets in a `list` or `dict` that does not perform to `DeviceMovable` protocol and is not a valid `torch.optim.Optimizer` will be returned without changing
        - device: A `torch.device` of target device
    - Returns: The same type of target but moved to target device
    """
    if isinstance(target, DeviceMovable):
        target = target.to(device)
    elif isinstance(target, torch.optim.Optimizer):
        for k, param in target.state.items():
            target.state[k] = move_to_device(param, device=device)
    elif isinstance(target, dict):
        target = {k: move_to_device(t, device) if isinstance(t, DeviceMovable) else t for k, t in target.items()}
    elif isinstance(target, Iterable):
        target = [move_to_device(t, device) if isinstance(t, DeviceMovable) or isinstance(t, torch.optim.Optimizer) else t for t in target]
    return target

def set_default(d: torch.device) -> None:
    if d.index is not None and d.type == "cuda":
        torch.cuda.set_device(d)