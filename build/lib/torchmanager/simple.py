# import typing modules
from __future__ import annotations
from typing import Any, Callable, List, Tuple, Union

# import required modules
import torch

class FunctionalModule(torch.nn.Module):
    '''
    A functional module that wraps a function into a `torch.nn.Module`
    
    - Properties:
        - module_list: A potential `torch.nn.ModuleList` that occurred in this module
    '''
    # properties
    _forward_fn: Callable[..., Any]
    module_list: torch.nn.ModuleList

    def __init__(self, forward_fn: Callable[..., Any], modules: List[torch.nn.Module]=[]) -> None:
        '''
        Constructor

        - parameters:
            - forward_fn: A `Callable` function
            - modules: A potential `list` of `torch.nn.Module` that appeared in `forward_fn`
        '''
        super().__init__()
        self._forward_fn = forward_fn
        self.module_list = torch.nn.ModuleList(modules)

    def forward(self, *args, **kwargs) -> Any:
        return self._forward_fn(*args, **kwargs)

class FunctionalNode:
    '''The basic functional node used to build modules for functional API'''
    layers: List[torch.nn.Module]

    def __init__(self) -> None:
        super().__init__()
        self.layers = []

    def __abs__(self) -> FunctionalNode:
        self.add_layer(FunctionalModule(lambda x: abs(x)))
        return self

    def __add__(self, other: Any) -> FunctionalNode:
        self.add_layer(FunctionalModule(lambda x: x + other))
        return self

    def __radd__(self, other: Any) -> FunctionalNode:
        self.add_layer(FunctionalModule(lambda x: other + x))
        return self

    def __mul__(self, other: Any) -> FunctionalNode:
        self.add_layer(FunctionalModule(lambda x: x * other))
        return self

    def __rmul__(self, other: Any) -> FunctionalNode:
        self.add_layer(FunctionalModule(lambda x: other * x))
        return self

    def __neg__(self) -> FunctionalNode:
        self.add_layer(FunctionalModule(lambda x: -x))
        return self

    def __pow__(self, other: Any) -> FunctionalNode:
        self.add_layer(FunctionalModule(lambda x: pow(x, other)))
        return self

    def __rpow__(self, other: Any) -> FunctionalNode:
        self.add_layer(FunctionalModule(lambda x: pow(other, x)))
        return self

    def __sub__(self, other: Any) -> FunctionalNode:
        self.add_layer(FunctionalModule(lambda x: x - other))
        return self

    def __rsub__(self, other: Any) -> FunctionalNode:
        self.add_layer(FunctionalModule(lambda x: other - x))
        return self

    def __truediv__(self, other: Any) -> FunctionalNode:
        self.add_layer(FunctionalModule(lambda x: x / other))
        return self

    def __rtruediv__(self, other: Any) -> FunctionalNode:
        self.add_layer(FunctionalModule(lambda x: other / x))
        return self

    def __matmul__(self, other: Any) -> FunctionalNode:
        self.add_layer(FunctionalModule(lambda x: x @ other))
        return self

    def __rmatmul__(self, other: Any) -> FunctionalNode:
        self.add_layer(FunctionalModule(lambda x: other @ x))
        return self

    def add_layer(self, layer: torch.nn.Module) -> None:
        self.layers.append(layer)

class BuiltModule(torch.nn.Module):
    '''
    A module that available to pass a `FunctionalNode` to built a FunctionalModule via functional api

    - Properties:
        - target: A `torch.nn.Module` to be built
    '''
    # properties
    target: torch.nn.Module

    def __init__(self, target: torch.nn.Module) -> None:
        super().__init__()
        self.target = target

    def forward(self, x: Any, *args, **kwargs) -> Any:
        if isinstance(x, FunctionalNode):
            x.add_layer(self.target)
            return x
        else: return self.target(x, *args, **kwargs)

class ModuleNotBuiltError(TypeError):
    pass

def build(target: Union[torch.nn.Module, Callable[..., Any]]) -> BuiltModule:
    '''
    Method to build a module via functional API

    - Parameters:
        - target: A `torch.nn.Module` to be built
    - Returns: A `BuiltModule` with the target module
    '''
    if isinstance(target, torch.nn.Module):
        return BuiltModule(target)
    else:
        return BuiltModule(FunctionalModule(target))

def functional(fn: Callable[[Any], Any]) -> FunctionalModule:
    '''
    A decorator wrapping function of functional

    - Parameters:
        fn: A `Callable` function that contains logics
    - Returns: A wrapped decorator `Callable` function
    '''
    # define wrapping function
    try:
        node = FunctionalNode()
        nodes: List[FunctionalNode] = list(fn(node))
        layers: List[torch.nn.Module] = []
        for n in nodes:
            layers.extend(n.layers)
        return FunctionalModule(fn, layers)
    except AttributeError or TypeError:
        raise ModuleNotBuiltError("[Functional Error]: One or more modules have not been built via build function.")

def multi_inputs_functional(num: int=1) -> Callable[[Callable[..., Any]], FunctionalModule]:
    def functional_wrap(fn: Callable[..., Any]) -> FunctionalModule:
        nodes = [FunctionalNode() for _ in range(num)]
        nodes = list(fn(*nodes))
        layers: List[torch.nn.Module] = []
        for n in nodes:
            layers.extend(n.layers)
        return FunctionalModule(fn, layers)
    return functional_wrap