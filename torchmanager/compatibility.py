from torchmanager_core import Version, deprecated
from torchmanager_core.typing import Any, Optional, TypeVar

from .metrics import Metric

T = TypeVar("T")


@deprecated("v1.3", "v1.4")
def convert(obj: T, from_version: Optional[Any] = None) -> T:
    """
    Convert an object from old version of torchmanager to the latest one.
    """
    # format version
    v = Version(from_version) if from_version is not None else Version("v1.0")

    # Return if None
    if isinstance(obj, Metric):
        obj.convert(from_version=v)
    return obj
