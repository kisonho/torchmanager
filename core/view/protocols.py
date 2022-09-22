from typing import Protocol
from enum import Enum

import abc

class VerboseControllable(Protocol):
    """The learning rate scheduler protocol"""
    @property
    @abc.abstractmethod
    def verbose(self) -> bool:
        raise NotImplementedError

    @verbose.setter
    @abc.abstractmethod
    def verbose(self, verbose: bool) -> None:
        raise NotImplementedError

class VerboseType(Enum):
    """Verbose type enum"""
    ALL = -1
    NONE = 0
    LOSS = 1
    METRICS = 2