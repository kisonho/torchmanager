from torchmanager_core import os, torch, _raise
from torchmanager_core.protocols import ModelContainer, MonitorType, StateDictLoadable
from torchmanager_core.typing import Any, Dict, Generic, Optional, TypeVar

from ..train import Checkpoint as Ckpt
from .callback import Callback

T = TypeVar('T', bound=StateDictLoadable)

class _Checkpoint(Callback, Generic[T]):
    """
    The callback to save the last checkpoint during training

    * extends: `.Callback

    - Properties:
        - ckpt_path: A `str` of checkpoint path
    """
    __ckpt_path: str
    _checkpoint: Ckpt[T]

    @property
    def ckpt_path(self) -> str:
        return self.__ckpt_path

    @ckpt_path.setter
    def ckpt_path(self, p: str) -> None:
        self.__ckpt_path = os.path.normpath(p)

    def __init__(self, model: T, ckpt_path: str, **kwargs: Any) -> None:
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

    def on_epoch_end(self, epoch: int, summary: Dict[str, float] = ..., val_summary: Optional[Dict[str, float]] = ...) -> None:
        self._checkpoint.save(epoch, self.ckpt_path)

class LastCheckpoint(_Checkpoint[T]):
    """
    Last checkpoint with frequency control support

    * extends: `_Checkpoint`
    
    - Properties:
        - freq: An `int` of checkpoint epoch frequency
    """
    __freq: int

    @property
    def freq(self) -> int:
        return self.__freq
    
    @freq.setter
    def freq(self, f: int) -> None:
        assert f > 0, _raise(ValueError(f"Frequency must be a positive number, got {f}. "))
        self.__freq = f

    def __init__(self, model: T, ckpt_path: str, freq: int = 1, **kwargs: Any) -> None:
        super().__init__(model, ckpt_path, **kwargs)
        self.freq = freq

    def on_epoch_end(self, epoch: int, summary: Dict[str, float] = ..., val_summary: Optional[Dict[str, float]] = ...) -> None:
        if epoch % self.freq == 0: super().on_epoch_end(epoch, summary, val_summary)

class BestCheckpoint(_Checkpoint[T]):
    """
    The callback to save the latest checkpoint for each epoch

    * extends: `_Checkpoint`

    - Properties:
        - best_score: A `float` of the best score to be monitored
        - monitor: A `str` of the summary name to be monitored
        - monitor_type: A `MonitorType` of the monitor
    """
    # properties
    best_score: float
    load_best: bool
    monitor: str
    monitor_type: MonitorType

    def __init__(self, monitor: str, model: T, ckpt_path: str, load_best: bool = False, monitor_type: MonitorType=MonitorType.MAX, **kwargs: Any) -> None:
        """
        Constructor

        - Parameters:
            - monitor: A `str` of monitored metric
            - monitor_type: A `MonitorType` of either `MIN` of `MAX` mode for the best model
        """
        super().__init__(model, ckpt_path, **kwargs)
        self.best_score = monitor_type.init_score
        self.load_best = load_best
        self.monitor = monitor
        self.monitor_type = monitor_type

    def on_epoch_end(self, epoch: int, summary: Dict[str, float] = ..., val_summary: Optional[Dict[str, float]] = ...) -> None:
        # get score
        score = val_summary[self.monitor] if val_summary is not None else summary[self.monitor]

        # save when best
        if score >= self.best_score and self.monitor_type == MonitorType.MAX:
            self.best_score = score
            super().on_epoch_end(epoch, summary, val_summary)
        elif score <= self.best_score and self.monitor_type == MonitorType.MIN:
            self.best_score = score
            super().on_epoch_end(epoch, summary, val_summary)

    def on_train_end(self, model: torch.nn.Module) -> None:
        # load best checkpoint
        if self.load_best:
            # load checkpoint
            best_ckpt: Ckpt[StateDictLoadable] = Ckpt.from_saved(self.ckpt_path)

            # load to model
            if isinstance(best_ckpt.model, ModelContainer):
                ckpt_model = best_ckpt.model.model
            else: ckpt_model = best_ckpt.model
            try: model.load_state_dict(ckpt_model.state_dict())
            except: raise TypeError(f"Reload best checkpoint to model failed: supposed to have {type(model)} in checkpoint, got {type(ckpt_model)}.")