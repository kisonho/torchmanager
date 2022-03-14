# import typing modules
from typing import Protocol, runtime_checkable
from enum import Enum

# import required modules
import abc, logging, warnings
from tqdm import tqdm

@runtime_checkable
class _VerboseControllable(Protocol):
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