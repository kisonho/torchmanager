from torchmanager_core import os, view
from torchmanager_core.typing import Generic, TypeVar, Union
from torchmanager_core.protocols import MonitorType, StateDictLoadable

from .callback import MultiCallbacks
from .ckpt import BestCheckpoint, LastCheckpoint
from .tensorboard import TensorBoard

T = TypeVar('T', bound=StateDictLoadable)


class Experiment(MultiCallbacks, Generic[T]):
    """
    The callback that wraps last and best checkpoints in `checkpoints` folder by `last.model` and `best_*.model` with tensorboard logs in `data` folder together into a wrapped *.exp file

    * extends: `.callback.Callback`
    * requires: `tensorboard` package
    """
    def __init__(self, experiment: str, model: T, monitors: Union[dict[str, MonitorType], list[str]] = {}, show_verbose: bool = True) -> None:
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
        if not experiment.endswith(".exp"):
            experiment += ".exp"
        experiment_dir = os.path.join("experiments", experiment)
        os.makedirs(experiment_dir, exist_ok=True)
        log_dir = os.path.join(experiment_dir, "data")
        ckpt_path = os.path.join(experiment_dir, "checkpoints")

        # initial checkpoints
        best_ckpts: list[BestCheckpoint] = []
        last_ckpt_path = os.path.join(ckpt_path, "last.model")
        last_ckpt = LastCheckpoint(model, last_ckpt_path)
        tensorboard = TensorBoard(log_dir)

        # initialize best checkpoints according to monitors
        monitors = monitors if isinstance(monitors, dict) else {monitor: MonitorType.MAX for monitor in monitors}
        for m, mode in monitors.items():
            best_ckpt_path = os.path.join(ckpt_path, f"best_{m}.model")
            best_ckpt = BestCheckpoint(m, model, best_ckpt_path, monitor_type=mode)
            best_ckpts.append(best_ckpt)

        # wrap callbacks
        super().__init__(last_ckpt, *best_ckpts, tensorboard)

        # initialize logging
        log_file = os.path.basename(experiment.replace(".exp", ".log"))
        log_path = os.path.join(experiment_dir, log_file)
        os.makedirs(experiment_dir, exist_ok=True)
        formatter = view.set_log_path(log_path)

        # initialize console
        if show_verbose:
            console = view.logging.StreamHandler()
            console.setLevel(view.logging.INFO)
            console.setFormatter(formatter)
            view.logger.addHandler(console)
