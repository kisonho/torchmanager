import torch
from typing import Callable


def add_gard_clip(model: torch.nn.Module, /, min_value: float, max_value: float, *, replace_nan: bool = False) -> None:
    """
    Add gradients clip for all trainable parameters to the target model

    - Parameters:
        - model: The target `torch.nn.Module` to add gradients clip
        - min_value: A `float` of the min gradients limit
        - max_value: A `float` of the max gradients limit
        - replace_nan: A `bool` flag of if replace nan values to zero in gradients
    """
    # initialize gradient clip function
    grad_clip_fn: Callable[[torch.Tensor], torch.Tensor] = lambda g: g.clamp(min_value, max_value).nan_to_num(0) if replace_nan else g.clamp(min_value, max_value)

    # loop for each parameter in model
    for param in model.parameters():
        param.register_hook(grad_clip_fn)


def backward_hook(model: torch.nn.Module, /, hook: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Add a backward hook function to all trainable parameters to the target model

    * Work as decorator:

    >>> model: torch.nn.Module = ...
    >>> @backward_hook(model)
    >>> def hook_fn(g: torch.Tensor) -> torch.Tensor:
    ...     ...

    - Parameters:
        - model: The target `torch.nn.Module` to add the hook function
        - hook: A hook function that accepts a `torch.Tensor` as gradients and returns the modified `torch.Tensor` gradients
    - Returns: the hook function registered
    """
    # loop for each parameter in model
    for param in model.parameters():
        param.register_hook(hook)
    return hook
