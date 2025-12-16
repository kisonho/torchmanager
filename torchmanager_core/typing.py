from typing import *  # type: ignore

from torch.nn import Module as _Module
from enum import Enum

Module = TypeVar("Module", bound=_Module)
SizedIterable = Collection
