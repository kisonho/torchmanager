from torchmanager_core import os
from torchmanager_core.typing import Dict, Generic, Optional, Set, Tuple, TypeVar

from ..train import Checkpoint
from ..train.protocols import StateDictLoadable
from .ckpt import BestCheckpoint, MonitorType
from .protocols import Frequency
from .tensorboard import TensorBoard

T = TypeVar('T', bound=StateDictLoadable)

class Experiment(TensorBoard, BestCheckpoint[T], Generic[T]):
    """
    The tensorboard callback that wraps last and best checkpoints in `checkpoints` folder by `last.model` and `best.model` with tensorboard logs in `data` folder together into a `*.exp` folder

    * extends: `.callback.Callback`, `.ckpt.BestCheckpoint`
    * requires: `tensorboard` package
    
    - Properties:
        - checkpoint: A `..train.Checkpoint` of target in `T`
        - path: A `str` of current experiment path
    """
    def __init__(self, experiment_path: str, model: T, monitor: str, monitor_type: MonitorType = MonitorType.MAX) -> None:
        experiment_path = os.path.normpath(experiment_path)
        log_dir = os.path.join(experiment_path, "data")
        ckpt_path = os.path.join(experiment_path, "checkpoints")
        TensorBoard.__init__(self, log_dir)
        BestCheckpoint.__init__(self, monitor, model, ckpt_path, monitor_type=monitor_type)

    def on_train_start(self, initial_epoch: int = 0) -> None:
        self.current_step = initial_epoch

    def step(self, summary: Dict[str, float], val_summary: Optional[Dict[str, float]] = None) -> Tuple[Set[str], dict[str, float], Optional[Dict[str, float]]]:
        # check if summary is given
        if self.freq == Frequency.EPOCH:
            # initialize score and checkpoint pathes
            score = val_summary[self.monitor] if val_summary is not None else summary[self.monitor]
            last_ckpt_path = os.path.join(self.ckpt_path, "last.model")
            best_ckpt_path = os.path.join(self.ckpt_path, "best.model")

            # save last checkpoints
            self._checkpoint.save(self.current_step, last_ckpt_path)

            # save best checkpoints
            if self.best_score <= score:
                self.best_score = score
                self._checkpoint.save(self.current_step, best_ckpt_path)

        # step to tensorboard
        return TensorBoard.step(self, summary, val_summary)