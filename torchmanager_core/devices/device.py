import torch, warnings
from typing import Any, Iterable, Optional, Type, TypeVar, Union, overload

from .protocols import DeviceMovable, DataParallelType

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
    GPUS: list[torch.device] = []

Module = TypeVar('Module', bound=torch.nn.Module)
P = TypeVar('P', bound=DataParallelType)


@overload
def data_parallel(raw_model: P, /, devices: list[torch.device] = GPUS, *, output_device: Optional[torch.device] = None, parallel_type: Type[P] = torch.nn.parallel.DataParallel) -> tuple[P, bool]:
    ...


@overload
def data_parallel(raw_model: Module, /, devices: list[torch.device] = GPUS, *, output_device: Optional[torch.device] = None, parallel_type: Type[P] = torch.nn.parallel.DataParallel) -> tuple[Union[Module, P], bool]:
    ...


def data_parallel(raw_model: Module, /, devices: list[torch.device] = GPUS, *, output_device: Optional[torch.device] = None, parallel_type: Type[P] = torch.nn.parallel.DataParallel) -> tuple[Union[Module, P], bool]:
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
    if GPU is not NotImplemented:
        torch.cuda.empty_cache()


def search(specified: Optional[Union[torch.device, list[torch.device]]] = None) -> tuple[torch.device, torch.device, list[torch.device]]:
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
            # assert all devices are the same type
            if d.type != specified[0].type:
                raise SystemError("All devices in the specified list must have the same device type.")

            # assert all devices have device index
            if d.index is None:
                raise SystemError("All devices in the specified list must have a device index.")
    else:
        raise SystemError("Specified device list must not be empty")
    return CPU, device, specified


@overload
def move_to_device(target: DeviceMovable, /, device: torch.device, *, recursive: bool = False) -> DeviceMovable:
    ...


@overload
def move_to_device(target: dict[str, Union[DeviceMovable, Any]], /, device: torch.device, *, recursive: bool = False) -> dict[str, Union[DeviceMovable, Any]]:
    ...


@overload
def move_to_device(target: list[Union[DeviceMovable, Any]], /, device: torch.device, *, recursive: bool = False) -> list[Union[DeviceMovable, Any]]:
    ...


def move_to_device(target: Union[DeviceMovable,  dict[str, Union[DeviceMovable,  Any]], list[Union[DeviceMovable,  Any]]], /, device: torch.device, *, recursive: bool = False) -> Union[DeviceMovable,  dict[str, Union[DeviceMovable,  Any]], list[Union[DeviceMovable,  Any]]]:
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
