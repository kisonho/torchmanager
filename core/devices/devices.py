import torch, warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

from .protocols import DeviceMovable

CPU = torch.device('cpu')
'''The main CPU'''

try:
    from torch.backends import mps
    METAL = torch.device('mps') if mps.is_available() else NotImplemented
    '''The overall Metal (Mac only) devices'''
except:
    METAL = NotImplemented

try:
    DEFAULT = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else CPU
    '''The default device'''
    GPU = torch.device('cuda') if torch.cuda.is_available() else NotImplemented
    '''The overall CUDA (NVIDIA only) devices'''
    GPUS = [torch.device(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    '''The list of available CUDA devices'''
except:
    DEFAULT = CPU
    GPU = NotImplemented
    GPUS = []

Module = TypeVar('Module', bound=torch.nn.Module)

def data_parallel(raw_model: Module, devices: List[torch.device] = GPUS, output_device: Optional[torch.device] = None, parallel_type: Type = torch.nn.parallel.DataParallel) -> Tuple[Union[Module, torch.nn.parallel.DataParallel], bool]:
    """
    Make a `torch.nn.Module` data parallel

    - Parameters:
        - raw_model: A target `torch.nn.Module`
        - devices: A `list` of target `torch.device`
        - output_device: An optional `torch.device` of the target output device for the paralleled model
        - parallel_type: A `type` of `torch.nn.parallel.DataParallel`
    - Returns: A `tuple` of either data paralleled `torch.nn.parallel.DataParallel` model if CUDA is available or a raw model if not, and a `bool` flag of if the model data paralleled successfuly.
    """
    if isinstance(raw_model, parallel_type):
        return raw_model, True
    elif GPU is not NotImplemented:
        device_ids = [d.index for d in devices]
        model = parallel_type(raw_model, device_ids=device_ids, output_device=output_device)
        return model, True
    else:
        warnings.warn(f"CUDA is not available, unable to use multi-GPUs.", ResourceWarning)
        return raw_model, False

def empty_cache() -> None:
    """Empty CUDA cache"""
    if GPU is not NotImplemented: torch.cuda.empty_cache()

def find(specified: Optional[torch.device] = None) -> Tuple[torch.device, torch.device]:
    """
    Find available devices

    - Pameters:
        - specified: An optional `torch.device` of specified
    - Returns: A `tuple` of CPU in `torch.device` and available device in `torch.device`
    """
    warnings.warn("This has been deprecated from v1.1.0 and will be removed in v1.2.0, use `torchmanager_core.devices.search` instead.", PendingDeprecationWarning)
    if specified is None and GPU is not NotImplemented:
        return (CPU, GPU)
    elif specified is None and METAL is not NotImplemented:
        return (CPU, METAL)
    elif specified is None:
        return (CPU, CPU)
    else: return CPU, specified

def search(specified: Optional[Union[torch.device, List[torch.device]]] = None) -> Tuple[torch.device, torch.device, List[torch.device]]:
    """
    Find available devices

    - Pameters:
        - specified: An optional `torch.device` of specified
    - Returns: A `tuple` of CPU in `torch.device`, available device in `torch.device` and `list` of specified devices in `torch.device`
    """
    if specified is None and GPU is not NotImplemented:
        return (CPU, GPU, GPUS)
    elif specified is None and METAL is not NotImplemented:
        return (CPU, METAL, [METAL])
    elif specified is None:
        return (CPU, CPU, [CPU])
    elif not isinstance(specified, list):
        device = specified
        specified = [specified]
    elif len(specified) > 0:
        # set default device
        device = torch.device(specified[0].type)

        # check for each device
        for d in specified:
            if d.type != GPU.type: raise SystemError("All devices in the specified list must have the same device type with GPU type")
            if d.index is None: raise SystemError("All devices in the specified list must have a device index")
    else: raise SystemError("Specified device list must not be empty")
    return CPU, device, specified

def move_to_device(target: Union[DeviceMovable,  Dict[str, Union[DeviceMovable,  Any]], List[Union[DeviceMovable,  Any]]], device: torch.device, recursive: bool = False) -> Union[DeviceMovable,  Dict[str, Union[DeviceMovable,  Any]], List[Union[DeviceMovable,  Any]]]:
    """
    Recurrently move a target variable to device if elements perform to `DeviceMovable` protocol
    
    - Parameters:
        - target: Either a target in `DeviceMovable`, a `dict` of targets in `DeviceMovable`, or a `list` of targets in `DeviceMovable`, targets in a `list` or `dict` that does not perform to `DeviceMovable` protocol will be returned without changing
        - device: A `torch.device` of target device
        - recursive: A `bool` flag of if move to device recursively
    - Returns: The same type of target but moved to target device
    """
    if isinstance(target, DeviceMovable):
        target = target.to(device)
    elif isinstance(target, dict):
        target = {k: move_to_device(t, device) if isinstance(t, DeviceMovable) or recursive else t for k, t in target.items()}
    elif isinstance(target, Iterable):
        target = [move_to_device(t, device) if isinstance(t, DeviceMovable) or recursive else t for t in target]
    return target

def set_default(d: torch.device) -> None:
    if d.index is not None and d.type == "cuda":
        torch.cuda.set_device(d)