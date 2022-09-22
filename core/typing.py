from torch.nn import Module as _Module
from typing import * # type: ignore
from enum import Enum

Module = TypeVar('Module', bound=_Module)
SizedIterable = Collection