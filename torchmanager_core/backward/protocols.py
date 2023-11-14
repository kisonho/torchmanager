import abc
from typing import Protocol


class Removable(Protocol):
    @abc.abstractmethod
    def remove(self) -> None:
        ...
