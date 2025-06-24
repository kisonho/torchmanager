import torch, warnings
from typing import Any, Iterable, Type, TypeVar, cast, overload

from .protocols import DeviceMovable, DataParallelType

CPU = torch.device('cpu')
'''The main CPU'''

# METAL is only available on macOS with Apple Silicon
if torch.backends.mps.is_available():
    METAL = torch.device('mps')
    '''The overall Metal (Mac only) devices'''
else:
    METAL = NotImplemented

# CUDA is only available on NVIDIA GPUs
if torch.cuda.is_available():
    DEFAULT = torch.device(torch.cuda.current_device())
    '''The default device'''
    GPU = torch.device('cuda')
    '''The overall CUDA (NVIDIA only) devices'''
    GPUS = [torch.device(i) for i in range(torch.cuda.device_count())]
    '''The list of available CUDA devices'''
else:
    DEFAULT = CPU
    GPU = NotImplemented
    GPUS: list[torch.device] = []

Module = TypeVar('Module', bound=torch.nn.Module)
P = TypeVar('P', bound=DataParallelType)

@overload
def data_parallel(raw_model: P, /, devices: list[torch.device] = GPUS, *, output_device: torch.device | None = None, parallel_type: Type[P] = torch.nn.parallel.DataParallel) -> tuple[P, bool]:
    ...

@overload
def data_parallel(raw_model: Module | P, /, devices: list[torch.device] = GPUS, *, output_device: torch.device | None = None, parallel_type: Type[P] = torch.nn.parallel.DataParallel) -> tuple[Module | P, bool]:
    ...

def data_parallel(raw_model: Module | P, /, devices: list[torch.device] = GPUS, *, output_device: torch.device | None = None, parallel_type: Type[P] = torch.nn.parallel.DataParallel) -> tuple[Module | P, bool]:
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
        raw_model = cast(P, raw_model)
        return raw_model, True
    elif GPU is not NotImplemented and not isinstance(raw_model, parallel_type):
        device_ids = [d.index for d in devices]
        model = parallel_type(raw_model, device_ids=device_ids, output_device=output_device)
        return model, True
    else:
        warnings.warn(f"CUDA is not available, unable to use multi-GPUs.", ResourceWarning)
        raw_model = cast(Module, raw_model)
        return raw_model, False

def empty_cache() -> None:
    """Empty CUDA cache"""
    if GPU is not NotImplemented:
        torch.cuda.empty_cache()

def search(specified: torch.device | list[torch.device] | None = None) -> tuple[torch.device, torch.device, list[torch.device]]:
    """
    Find available devices

    - Pameters:
        - specified: An optional `torch.device` of specified
    - Returns: A `tuple` of CPU in `torch.device`, available device in `torch.device` and `list` of specified devices in `torch.device`
    """
    if specified is None and GPU is not NotImplemented:  # automatically search for CUDA devices
        return (CPU, GPU, GPUS)
    elif specified is None and METAL is not NotImplemented:  # automatically search for METAL devices
        return (CPU, METAL, [METAL])
    elif specified is None:  # no available acceleration devices, return CPU
        return (CPU, CPU, [CPU])
    elif not isinstance(specified, list):  # if specified is a single device
        device = specified
        specified = [specified]
    elif len(specified) > 0:  # if specified is a list of devices
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
    else:  # if specified is an empty list
        raise SystemError("Specified device list must not be empty")
    return CPU, device, specified

T = TypeVar('T', bound=DeviceMovable | dict[str, DeviceMovable | Any] | list[DeviceMovable | Any])

def move_to_device(target: T, /, device: torch.device, *, recursive: bool = True) -> T:
    """
    Recurrently move a target variable to device if elements perform to `DeviceMovable` protocol

    - Parameters:
        - target: Either a target in `DeviceMovable`, a `dict` of targets in `DeviceMovable`, or a `list` of targets in `DeviceMovable`, targets in a `list` or `dict` that does not perform to `DeviceMovable` protocol will be returned without changing
        - device: A `torch.device` of target device
        - recursive: A `bool` flag of if move to device recursively
    - Returns: The same type of target but moved to target device
    """
    if isinstance(target, DeviceMovable):  # if target performs `DeviceMovable` protocol
        target = target.to(device)
    elif isinstance(target, dict):  # if target is a dict
        target = cast(T, {k: move_to_device(t, device) if isinstance(t, DeviceMovable) or recursive else t for k, t in target.items()})
    elif isinstance(target, Iterable):  # if target is an iterable
        target = cast(T, [move_to_device(t, device) if isinstance(t, DeviceMovable) or recursive else t for t in target])
    return target

def set_default(d: torch.device) -> None:
    if d.index is not None and d.type == "cuda":
        torch.cuda.set_device(d)
