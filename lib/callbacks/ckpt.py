from __future__ import annotations
from typing import Any
from ..core import os, sys, view
from ..core.typing import Dict, Enum, Generic, Optional, TypeVar
from ..train.checkpoint import Checkpoint as Ckpt
from .callback import Callback

T = TypeVar('T')

class LastCheckpoint(Callback, Generic[T]):
    """
    The callback to save the last checkpoint during training

    - Properties:
        - ckpt_path: A `str` of checkpoint path
    """
    _checkpoint: Ckpt[T]
    ckpt_path: str

    def __init__(self, model: Any, ckpt_path: str, **kwargs) -> None:
        """
        Constructor

        - Parameters:
            - model: Any type of model to be saved
            - ckpt_path: A `str` of the checkpoint path
            - **kwargs: Other arguments in `Checkpoint` constructor
        """
        super().__init__()
        self._checkpoint = Ckpt(model, **kwargs)
        self.ckpt_path = os.path.normpath(ckpt_path)

    def on_epoch_end(self, epoch: int, *args, **kwargs) -> None:
        self._checkpoint.save(epoch, self.ckpt_path)

class Checkpoint(LastCheckpoint, Generic[T]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        view.warnings.warn("[Deprecation Warning]: Checkpoint callback has been renamed to LastCheckpoint and was deprecated from v 1.0.0, it will be removed at v1.1.0.", DeprecationWarning)

class MonitorType(Enum):
    """The enum of monitor types"""
    MIN = int(0)
    MAX = int(1)

    @property
    def init_score(self) -> float:
        if self == MonitorType.MAX:
            return -1
        elif self == MonitorType.MIN:
            return sys.float_info.max
        else:
            raise ValueError(f'[MonitorType Error]: Monitor type {self} is not supported.')

class BestCheckpoint(LastCheckpoint, Generic[T]):
    """
    The callback to save the latest checkpoint for each epoch

    - Properties:
        - best_score: A `float` of the best score to be monitored
        - monitor: A `str` of the summary name to be monitored
        - monitor_type: A `MonitorType` of the monitor
    """
    # properties
    best_score: float
    monitor: str
    monitor_type: MonitorType

    def __init__(self, monitor: str, *args, monitor_type: MonitorType=MonitorType.MAX, **kwargs) -> None:
        """
        Constructor

        - Parameters:
            - monitor: A `str` of monitored metric
            - monitor_type: A `MonitorType` of either `MIN` of `MAX` mode for the best model
        """
        super().__init__(*args, **kwargs)
        self.monitor = monitor
        self.monitor_type = monitor_type
        self.best_score = monitor_type.init_score

    def on_epoch_end(self, *args, summary: Dict[str, float]={}, val_summary: Optional[Dict[str, float]]=None, **kwargs) -> None:
        # get score
        score = val_summary[self.monitor] if val_summary is not None else summary[self.monitor]

        # save when best
        if score >= self.best_score and self.monitor_type == MonitorType.MAX:
            self.best_score = score
            super().on_epoch_end(*args, **kwargs)
        elif score <= self.best_score and self.monitor_type == MonitorType.MIN:
            self.best_score = score
            super().on_epoch_end(*args, **kwargs)