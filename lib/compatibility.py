from torchmanager_core.typing import Any, Optional, TypeVar

from .losses import KLDiv
from .metrics import Accuracy, MAE, Reduction

T = TypeVar("T")


def convert(obj: T, from_version: Optional[str] = None) -> T:
    """
    Convert an object from old version of torchmanager to the latest one.
    """
    # Return if None
    if obj is None:
        return obj

    if from_version is None: # convert from 1.0 (from_version = None)
        # convert KLDiv
        if isinstance(obj, KLDiv) and not hasattr(obj, "replace_nan"):
            obj.replace_nan = False
        elif isinstance(obj, KLDiv) and not hasattr(obj, "_t"):
            obj._t = None

        # convert Accuracy and MAE
        if isinstance(obj, Accuracy) and isinstance(obj, MAE) and not hasattr(obj, "reduction"):
            obj.reduction = Reduction.MEAN
    return obj