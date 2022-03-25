from torch.nn import Module
from typing import * # type: ignore
from enum import Enum

Module = TypeVar('Module', bound=Module)

@runtime_checkable
class SizedIterable(Protocol):
    def __len__(self) -> int:
        return NotImplemented

    def __iter__(self) -> Any:
        return NotImplemented