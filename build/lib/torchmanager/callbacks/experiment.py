from torchmanager_core import os
from torchmanager_core.typing import Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union
from torchmanager_core.view import logging

from ..train.protocols import StateDictLoadable
from .ckpt import _Checkpoint, MonitorType
from .tensorboard import TensorBoard

T = TypeVar('T', bound=StateDictLoadable)

class Experiment(TensorBoard, _Checkpoint[T], Generic[T]):
    """
    The tensorboard callback that wraps last and best checkpoints in `checkpoints` folder by `last.model` and `best_*.model` with tensorboard logs in `data` folder together into a `*.exp` folder

    * extends: `.callback.Callback`, `.ckpt.BestCheckpoint`
    * requires: `tensorboard` package
    """
    __monitors: Dict[str, MonitorType]
    best_scores: Dict[str, Optional[float]]

    @property
    def monitors(self) -> Dict[str, MonitorType]: return self.__monitors

    def __init__(self, experiment_dir: str, model: T, monitors: Union[Dict[str, MonitorType], List[str]]={}, show_verbose: bool = True) -> None:
        """
        Constructor

        - Parameters:
            - experiment_dir: A `str` of target folder for the experiment
            - model: A target model to be tracked during experiment in `T`
            - monitors: A `list` of metric name if all monitors are using `MonitorType.MAX` to track, or `dict` of metric name to be tracked for the best checkpoint in `str` and the `.ckpt.MonitorType` to track as values
            - show_verbose: A `bool` flag of if showing loggins in console
        """
        # call super constructor
        experiment_dir = os.path.normpath(experiment_dir)
        if not experiment_dir.endswith(".exp"): experiment_dir += ".exp"
        os.makedirs(experiment_dir, exist_ok=True)
        log_dir = os.path.join(experiment_dir, "data")
        ckpt_path = os.path.join(experiment_dir, "checkpoints")
        TensorBoard.__init__(self, log_dir)
        _Checkpoint.__init__(self, model, ckpt_path)
        self.__monitors = monitors if isinstance(monitors, dict) else {monitor: MonitorType.MAX for monitor in monitors}

        # initialize scores
        for m in self.monitors:
            self.best_scores[m] = None

        # initialize logging
        log_file = os.path.basename(experiment_dir.replace(".exp", ".log"))
        log_path = os.path.join(experiment_dir, log_file)
        logging.basicConfig(level=logging.INFO, filename=log_path, format="%(message)s")

        # initialize console
        if show_verbose:
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            console.setFormatter(formatter)
            logging.getLogger().addHandler(console)

    def on_train_start(self, initial_epoch: int = 0) -> None:
        self.current_step = initial_epoch

    def step(self, summary: Dict[str, float], val_summary: Optional[Dict[str, float]] = None) -> Tuple[Set[str], dict[str, float], Optional[Dict[str, float]]]:
        # save last checkpoints
        last_ckpt_path = os.path.join(self.ckpt_path, "last.model")
        self._checkpoint.save(self.current_step, last_ckpt_path)

        # loop for monitors
        for monitor, monitor_type in self.monitors.items():
            # initialize checkpoint path and score
            best_ckpt_path = os.path.join(self.ckpt_path, f"best_{monitor}.model")
            score = val_summary[monitor] if val_summary is not None else summary[monitor]

            # save best checkpoint
            if self.best_scores[monitor] is None:
                self.best_score = score
                self._checkpoint.save(self.current_step, best_ckpt_path)
            elif self.best_score <= score and monitor_type == MonitorType.MAX:
                self.best_score = score
                self._checkpoint.save(self.current_step, best_ckpt_path)
            elif self.best_score >= score and monitor_type == MonitorType.MIN:
                self.best_score = score
                self._checkpoint.save(self.current_step, best_ckpt_path)

        # step to tensorboard
        return TensorBoard.step(self, summary, val_summary)