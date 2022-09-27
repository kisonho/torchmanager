from typing import Protocol, runtime_checkable
from enum import Enum

@runtime_checkable
class VerboseControllable(Protocol):
    """The protocol which contains the `verbose` property"""
    verbose: bool

class VerboseType(Enum):
    """Verbose type enum"""
    ALL = -1
    NONE = 0
    LOSS = 1
    METRICS = 2