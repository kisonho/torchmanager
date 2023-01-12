from torchmanager_core import os
from torchmanager_core.typing import Any, Dict, Generic, List, TypeVar, Union
from torchmanager_core.protocols import MonitorType, StateDictLoadable
from torchmanager_core.view import logging

from .callback import Callback
from .ckpt import BestCheckpoint, LastCheckpoint
from .tensorboard import TensorBoard

T = TypeVar('T', bound=StateDictLoadable)

class Experiment(Callback, Generic[T]):
    """
    The tensorboard callback that wraps last and best checkpoints in `checkpoints` folder by `last.model` and `best_*.model` with tensorboard logs in `data` folder together into a wrapped *.exp file

    * extends: `.callback.Callback`
    * requires: `tensorboard` package

    - Properties:
        - best_ckpts: A `list` of `.ckpt.BestCheckpoint` callbacks that records best checkpoints
        - last_ckpt: A `.ckpt.LastCheckpoint` callback that records the last checkpoint
        - tensorboard: A `.ckpt.TensorBoard` callback that records data to tensorboard
    """
    best_ckpts: List[BestCheckpoint[T]]
    last_ckpt: LastCheckpoint[T]
    tensorboard: TensorBoard

    def __init__(self, experiment: str, model: T, monitors: Union[Dict[str, MonitorType], List[str]]={}, show_verbose: bool = True) -> None:
        """
        Constructor

        - Parameters:
            - experiment: A `str` of target folder for the experiment
            - model: A target model to be tracked during experiment in `T`
            - monitors: A `list` of metric name if all monitors are using `MonitorType.MAX` to track, or `dict` of metric name to be tracked for the best checkpoint in `str` and the `.ckpt.MonitorType` to track as values
            - show_verbose: A `bool` flag of if showing loggins in console
        """
        # call super constructor
        experiment = os.path.normpath(experiment)
        if not experiment.endswith(".exp"): experiment += ".exp"
        os.makedirs(experiment, exist_ok=True)
        log_dir = os.path.join(experiment, "data")
        ckpt_path = os.path.join(experiment, "checkpoints")
        
        # initial checkpoints
        self.best_ckpts = []
        last_ckpt_path = os.path.join(ckpt_path, "last.model")
        self.last_ckpt = LastCheckpoint(model, last_ckpt_path)
        self.tensorboard = TensorBoard(log_dir)

        # initialize best checkpoints according to monitors
        monitors = monitors if isinstance(monitors, dict) else {monitor: MonitorType.MAX for monitor in monitors}
        for m, mode in monitors.items():
            best_ckpt_path = os.path.join(ckpt_path, f"best_{m}.model")
            best_ckpt = BestCheckpoint(m, model, best_ckpt_path, monitor_type=mode)
            self.best_ckpts.append(best_ckpt)

        # initialize logging
        log_file = os.path.basename(experiment.replace(".exp", ".log"))
        log_path = os.path.join(experiment, log_file)
        logger = logging.getLogger("torchmanager")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # initialize console
        if show_verbose:
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(formatter)
            logger.addHandler(console)

    def on_train_start(self, initial_epoch: int = 0) -> None:
        for best_ckpt in self.best_ckpts:
            best_ckpt.on_batch_start(initial_epoch)
        self.last_ckpt.on_train_start(initial_epoch)
        self.tensorboard.on_train_start(initial_epoch)

    def on_batch_end(self, *args: Any, **kwargs: Any) -> None:
        for best_ckpt in self.best_ckpts:
            best_ckpt.on_batch_end(*args, **kwargs)
        self.last_ckpt.on_batch_end(*args, **kwargs)
        self.tensorboard.on_batch_end(*args, **kwargs)

    def on_batch_start(self, *args: Any, **kwargs: Any) -> None:
        for best_ckpt in self.best_ckpts:
            best_ckpt.on_batch_start(*args, **kwargs)
        self.last_ckpt.on_batch_start(*args, **kwargs)
        self.tensorboard.on_batch_start(*args, **kwargs)

    def on_epoch_end(self, *args: Any, **kwargs: Any) -> None:
        for best_ckpt in self.best_ckpts:
            best_ckpt.on_epoch_end(*args, **kwargs)
        self.last_ckpt.on_epoch_end(*args, **kwargs)
        self.tensorboard.on_epoch_end(*args, **kwargs)

    def on_epoch_start(self, *args: Any, **kwargs: Any) -> None:
        for best_ckpt in self.best_ckpts:
            best_ckpt.on_epoch_start(*args, **kwargs)
        self.last_ckpt.on_epoch_start(*args, **kwargs)
        self.tensorboard.on_epoch_start(*args, **kwargs)