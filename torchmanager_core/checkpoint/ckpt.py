import os, torch
from torchmanager_core import view
from torchmanager_core.protocols import ModelContainer, StateDictLoadable, WrappedFn
from torchmanager_core.typing import Any, Generic, Optional, OrderedDict, TypeVar

T = TypeVar("T", bound=StateDictLoadable)


class Checkpoint(Generic[T]):
    """
    The callback to save the latest checkpoint for each epoch

    - Properties:
        - last_epoch: An `int` of the last epoch index
        - loss_fn: An optional `Loss` object
        - metrics: An optional `dict` with name in `str` and value of the `Metric` objects for metrics
        - model: Any type of model to be saved
        - optimizer: A `torch.nn.Optimizer` to be saved
        - save_weights_only: A `bool` flag of if only save state_dict of model
    """

    last_epoch: int
    loss_fn: Optional[StateDictLoadable]
    metrics: dict[str, StateDictLoadable]
    model: T
    optimizer: Optional[torch.optim.Optimizer]
    save_weights_only: bool

    def __init__(self, model: T, last_epoch: int = 0, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[StateDictLoadable] = None, metrics: Optional[dict[str, StateDictLoadable]] = None, save_weights_only: bool = False) -> None:
        """
        Constructor

        - Parameters:
            - model: Any type of model to be saved
            - epoch: An `int` of epoch index
            - optimizer: An optional `torch.optim.Optimizer` to be recorded
            - loss_fn: An optional `torch.nn.Module` for loss function to be recorded
            - metrics: An optional `dict` of the metrics with key in `str` and value in `torch.nn.Module` to be recorded
        """
        super().__init__()
        self.last_epoch = last_epoch
        self.metrics = {}
        self.model = model
        self.optimizer = optimizer
        self.save_weights_only = save_weights_only

        # check if model is a container
        if isinstance(model, ModelContainer):
            # check if loss function is wrapped
            if isinstance(model.loss_fn, WrappedFn):
                view.warnings.warn("Loss function is wrapped and is not supported by checkpoint, use `losses.Loss` instead.")
            
            # check if metrics is wrapped
            for name, metric in model.metric_fns.items():
                if isinstance(metric, WrappedFn):
                    view.warnings.warn(f"Metric {name} is wrapped and is not supported by checkpoint, use `metrics.Metric` instead.")

        # check if loss function is wrapped
        if isinstance(loss_fn, WrappedFn):
            view.warnings.warn("Loss function is wrapped and is not supported by checkpoint, use `losses.Loss` instead.")
            self.loss_fn = None
        else:
            self.loss_fn = loss_fn

        # check if metrics is given
        if metrics is not None:
            # check if metrics is wrapped
            for name, metric in self.metrics.items():
                if isinstance(metric, WrappedFn):
                    view.warnings.warn(f"Metric {name} is wrapped and is not supported by checkpoint, use `metrics.Metric` instead.")
                    continue
                else:
                    self.metrics[name] = metric

    @classmethod
    def from_saved(cls, ckpt_path: str, map_location: Optional[torch.device] = None, model: Optional[StateDictLoadable] = None):
        """
        Load checkpoint from a saved checkpoint file

        - Parameters:
            - ckpt_path: A `str` of file path
            - model: Any object to be loaded
        """
        # load checkpint dictionary
        ckpt: dict[str, Any] = torch.load(ckpt_path, map_location=map_location)

        # load model
        if model is not None and ckpt["save_weights_only"] is True:
            state_dict: OrderedDict[str, Any] = ckpt["model"]
            model.load_state_dict(state_dict=state_dict)
            ckpt["model"] = model
        elif model is None and ckpt["save_weights_only"] is True:
            raise TypeError("Model must be given to load this checkpoint because `save_weights_only` was set to be `True`.")
        else:
            # remove data parallel wrap
            if isinstance(ckpt["model"], torch.nn.parallel.DataParallel):
                ckpt["model"] = ckpt["model"].module

            # load model structure with checkpoint weights
            if model is not None:
                saved_model: StateDictLoadable = ckpt["model"]  # type: ignore
                model.load_state_dict(state_dict=OrderedDict(saved_model.state_dict()))
                ckpt["model"] = model
        return cls(**ckpt)

    def save(self, epoch: int, ckpt_path: str) -> None:
        """
        Saves current checkpoint

        - Parameters:
            - epoch: The `int` index of current epoch to save
            - ckpt_path: The `str` of checkpoint path to save
        """
        # update last epoch and fetch current checkpoint
        self.last_epoch = epoch

        # initialize checkpoint dictionary
        ckpt = self.__dict__

        # check if save weights only
        if self.save_weights_only is True:
            model = self.model.module if isinstance(self.model, torch.nn.parallel.DataParallel) else self.model
            ckpt["model"] = model.state_dict()

        # remove data parallel wrap
        if isinstance(ckpt["model"], torch.nn.parallel.DataParallel):
            ckpt["model"] = ckpt["model"].module

        # check if model is a container
        if isinstance(ckpt["model"], ModelContainer):
            # initialize loss
            loss_fn = ckpt["model"].loss_fn

            # check if loss in container is wrapped
            if isinstance(loss_fn, WrappedFn):
                ckpt["model"].loss_fn = None

            # initialize metrics
            full_metrics = ckpt["model"].metric_fns
            supported_metrics: dict[str, StateDictLoadable] = {}

            # check if full metrics contains wrapped function
            for name, metric in full_metrics.items():
                if isinstance(metric, WrappedFn):
                    continue
                else:
                    supported_metrics[name] = metric

            # change metrics in container to supported metrics
            ckpt["model"].metric_fns = supported_metrics
        else:
            loss_fn = ckpt["loss_fn"]
            full_metrics = ckpt["metrics"]

        # save
        ckpt_path = os.path.normpath(ckpt_path)
        ckpt_dir = os.path.dirname(ckpt_path)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(ckpt, ckpt_path)

        # recover loss and metrics in container
        if isinstance(ckpt["model"], ModelContainer):
            ckpt["model"].loss_fn = loss_fn
            ckpt["model"].metric_fns = full_metrics


def list_checkpoints(experiment_dir: str) -> list[str]:
    """
    List all checkpoints in the given experiment folder

    - Parameters:
        - experiment_dir: A `str` of experiment directory
    - Returns: A `list` of checkpoints name in the experiment
    """
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    return [file.replace(".model", "") for file in os.listdir(ckpt_dir) if file.endswith(".model")]


def load(experiment_dir: str, ckpt_name: str) -> Checkpoint[Any]:
    """
    Load a checkpoint by its name and the experiment folder

    - Parameters:
        - experiment_dir: A `str` of experiment directory
        - ckpt_name: A `str` of the checkpoint name in the directory
    - Returns: The loaded `Checkpoint`
    """
    if not ckpt_name.endswith(".model"):
        ckpt_name += ".model"
    return Checkpoint.from_saved(os.path.join(experiment_dir, "checkpoints", ckpt_name))
