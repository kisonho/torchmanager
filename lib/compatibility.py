from torchmanager_core.typing import Any, Optional, TypeVar

from .basic import BaseManager
from .losses import KLDiv
from .metrics import Accuracy, MAE, Reduction
from .training import Manager as TrainingManager

M = TypeVar("M", bound=BaseManager)
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
    elif from_version < 'v1.2': # convert from 1.1
        if isinstance(obj, TrainingManager):
            obj.clip_gradients = False

    # convert manager
    if isinstance(obj, BaseManager):
        obj.model = convert(obj.model, from_version=from_version)
        obj.loss_fn = convert(obj.loss_fn, from_version=from_version) # type: ignore
        for k in obj.metric_fns:
            obj.metric_fns[k] = convert(obj.metric_fns[k], from_version=from_version)
    return obj