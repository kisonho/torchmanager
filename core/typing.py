from typing import *  # type: ignore

from torch.nn import Module as _Module
from typing_extensions import Self
from enum import Enum

Module = TypeVar("Module", bound=_Module)
SizedIterable = Collection
