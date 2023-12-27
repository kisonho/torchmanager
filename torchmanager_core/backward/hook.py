import torch
from typing import Callable, Union, overload

from .protocols import Removable


def add_gard_clip(model: torch.nn.Module, /, min_value: float, max_value: float, *, replace_nan: bool = False) -> list[Removable]:
    """
    Add gradients clip for all trainable parameters to the target model

    - Parameters:
        - model: The target `torch.nn.Module` to add gradients clip
        - min_value: A `float` of the min gradients limit
        - max_value: A `float` of the max gradients limit
        - replace_nan: A `bool` flag of if replace nan values to zero in gradients
    - Returns: a list of `Removable` hook handlers
    """
    # initialize gradient clip function
    grad_clip_fn: Callable[[torch.Tensor], torch.Tensor] = lambda g: g.clamp(min_value, max_value).nan_to_num(0) if replace_nan else g.clamp(min_value, max_value)
    handlers: list[Removable] = []

    # loop for each parameter in model
    for param in model.parameters():
        hook_handler = param.register_hook(grad_clip_fn)
        handlers.append(hook_handler)
    return handlers


@overload
def backward_hook(model: torch.nn.Module, /, hook: Callable[[torch.Tensor], torch.Tensor], *, as_decorator: bool = True) -> Callable[[torch.Tensor], torch.Tensor]:
    ...


@overload
def backward_hook(model: torch.nn.Module, /, hook: Callable[[torch.Tensor], torch.Tensor], *, as_decorator: bool = False) -> list[Removable]:
    ...


def backward_hook(model: torch.nn.Module, /, hook: Callable[[torch.Tensor], torch.Tensor], *, as_decorator: bool = True) -> Union[Callable[[torch.Tensor], torch.Tensor], list[Removable]]:
    """
    Add a backward hook function to all trainable parameters to the target model

    * Work as decorator:

    >>> model: torch.nn.Module = ...
    >>> @backward_hook(model)
    >>> def hook_fn(g: torch.Tensor) -> torch.Tensor:
    ...     ...
    >>> assert isinstance(hook_fn, Callable)

    * Or work as function:

    >>> model: torch.nn.Module = ...
    >>> hook_fn: Callable[[torch.Tensor], torch.Tensor] = ...
    >>> handlers = backward_hook(model, hook=hook_fn, as_decorator=False)
    >>> assert isinstance(handlers, list)

    - Parameters:
        - model: The target `torch.nn.Module` to add the hook function
        - hook: A hook function that accepts a `torch.Tensor` as gradients and returns the modified `torch.Tensor` gradients
    - Returns: the hook function registered if used as a decorator, otherwise a list of `Removable` hook handlers
    """
    # initialize
    handlers: list[Removable] = []

    # loop for each parameter in model
    for param in model.parameters():
        handler = param.register_hook(hook)
        handlers.append(handler)
    return hook if as_decorator else handlers
