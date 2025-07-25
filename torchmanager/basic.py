from torch.optim.optimizer import Optimizer
from torchmanager_core import checkpoint, devices, errors, torch, Version, API_VERSION
from torchmanager_core.protocols import Resulting
from torchmanager_core.typing import Any, Collection, Generic, Mapping, Module, OrderedDict, Self, cast

from .losses import Loss, MultiLosses, ParallelLoss
from .metrics import Metric


class BaseManager(Generic[Module]):
    """
    The basic manager

    * Implements: `torchmanager_core.devices.DeviceMovable`, `.callbacks.protocols.ModelContainer`

    Compile a model, optimizer, loss function, and metrics into the manager:
    >>> import torch
    >>> from torchmanager import losses, metrics
    >>> class SomeModel(torch.nn.Module): ...
    >>> model = SomeModel()
    >>> optimizer = torch.optim.SGD(...)
    >>> loss_fn = losses.Loss(...)
    >>> metric_fns = {
    ...     'metric_1': ...,
    ...     'metric_2': ...,
    ... }
    >>> manager = Manager(model, optimizer, loss_fn, metric_fns=metric_fns)

    Accepts multi losses as dictionary:
    >>> loss_fn = {
    ...     'loss_1': ...,
    ...     'loss_2': ...,
    ... }
    >>> manager = Manager(..., loss_fn=loss_fn)

    - Properties:
        - compiled: A `bool` flag of if the manager has been compiled
        - device: A `torch.device` of the current running device
        - loss_dict: A `dict` of loss functions with their names in `str`
        - loss_fn: An optional `Loss` for the objective function
        - metrics: A `dict` of metrics with a name in `str` as keys and a `Metric` as values
        - model: A target `torch.nn.Module` to be trained
        - optimizer: A `Optimizer` to train the model
        - raw_loss_fn: An optional `Loss` of the non-paralleled loss function
        - raw_model: A non-paralleled target `torch.nn.Module` model
    """
    # properties
    loss_fn: Resulting | None
    """The optional main loss function in `Resulting`"""
    metric_fns: dict[str, Resulting]
    """A `dict` of the metric functions with names as keys in `str` and metric functions as values in `torch.metrics.Metric`"""
    model: Module | torch.nn.DataParallel
    optimizer: Optimizer | None
    version: Version

    @property
    def compiled(self) -> bool:
        """The `bool` flag of if this manager has been compiled for training"""
        return True if self.loss_fn is not None and self.optimizer is not None else False

    @property
    def loss_dict(self) -> dict[str, Resulting]:
        """The `dict` of loss functions with their names in `str`"""
        loss_dict = {"loss": self.loss_fn} if self.loss_fn is not None else {}
        losses_in_metrics = {k: v for k, v in self.metric_fns.items() if isinstance(v, Loss)}
        loss_dict.update(losses_in_metrics)
        return loss_dict

    @loss_dict.setter
    def loss_dict(self, loss_dict: dict[str, Loss]) -> None:
        """Set the loss functions with their names in `str`"""
        # initialize loss
        loss_fn_mapping = {f"{name}_loss": fn for name, fn in loss_dict.items()}
        self.metric_fns.update(loss_fn_mapping)
        loss_fn = MultiLosses([l for l in loss_fn_mapping.values()])
        self.loss_fn = loss_fn

    @property
    def raw_loss_fn(self) -> Resulting | None:
        """The `torchmanager.losses.Loss` controlled by this manager without `torch.nn.DataParallel` wrap"""
        if self.loss_fn is None:
            return self.loss_fn
        elif isinstance(self.loss_fn, ParallelLoss):
            return self.loss_fn.module
        elif isinstance(self.loss_fn._metric_fn, torch.nn.parallel.DataParallel):
            self.loss_fn._metric_fn = self.loss_fn._metric_fn.module
            return self.loss_fn
        else:
            return self.loss_fn

    @property
    def raw_model(self) -> Module:
        """The `Module` controlled by this manager without `torch.nn.DataParallel` wrap"""
        return cast(Module, self.model.module) if isinstance(self.model, torch.nn.DataParallel) else self.model

    def __init__(self, model: Module, optimizer: Optimizer | None = None, loss_fn: Loss | dict[str, Loss] | None = None, metrics: dict[str, Metric] = {}) -> None:
        """
        Constructor

        - Parameters:
            - loss_fn: An optional `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: An optional `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
            - model: An optional target `torch.nn.Module` to be trained
            - optimizer: An optional `Optimizer` to train the model
        """
        # initialize
        self.metric_fns = {}
        self.model = model
        self.version = API_VERSION

        # initialize loss
        if isinstance(loss_fn, dict):
            loss_fn_mapping: dict[str, Loss] = {f"{name}_loss": fn for name, fn in loss_fn.items()}
            self.metric_fns.update(loss_fn_mapping)
            loss_fn = MultiLosses([l for l in loss_fn_mapping.values()])
        self.loss_fn = loss_fn

        # initialize metrics
        for name, fn in metrics.items():
            name = f"{name}_loss" if isinstance(fn, Loss) and not name.endswith("_loss") else name
            self.metric_fns[name] = fn

        # initialize optimizer
        self.optimizer = optimizer

        # reset to CPU
        self.reset()

    def __enter__(self) -> Self:
        """Enter the training mode"""
        # initialize status
        self.model = self.model.train()
        if self.loss_fn is not None:
            self.loss_fn = self.loss_fn.train()
        for k, m in self.metric_fns.items():
            self.metric_fns[k] = m.train()
        self.reset_metrics()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """Exit the training mode"""
        # finalize status
        self.model = self.model.eval()
        if self.loss_fn is not None:
            self.loss_fn = self.loss_fn.eval()
        for k, m in self.metric_fns.items():
            self.metric_fns[k] = m.eval()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} (version={self.version})"

    def convert(self) -> None:
        """
        Convert from a version to current version

        - Parameters:
            - from_version: A `torchmanager.version.Version` to convert from
        """
        # get from version
        from_version = self.version

        # check manager version
        if from_version < API_VERSION:
            # convert loss version
            if self.raw_loss_fn is not None:
                self.raw_loss_fn.convert(from_version)
        
            # convert metrics version
            for m in self.metric_fns.values():
                m.convert(from_version)

            # set version
            self.version = API_VERSION

    def data_parallel(self, target_devices: list[torch.device]) -> bool:
        """
        Data parallel all available models

        - Parameters:
            - target_devices: The target multiple devices for data parallel
        - Returns: A `bool` flag of if use multi GPUs
        """
        # multi gpus support for loss
        if self.loss_fn is not None:
            assert isinstance(self.raw_loss_fn, Loss), errors._raise(TypeError("The loss function is not a valid `Loss` object."))
            paralleled_loss_fn, use_multi_gpus = devices.data_parallel(self.raw_loss_fn, devices=target_devices, parallel_type=ParallelLoss)
            assert isinstance(paralleled_loss_fn, ParallelLoss) or isinstance(paralleled_loss_fn, Loss), errors._raise(TypeError("Paralleled function is not a valid `ParallelLoss` or `Loss` after parallel."))
            self.loss_fn = paralleled_loss_fn

        # multi gpus support for model
        self.model, use_multi_gpus = devices.data_parallel(self.raw_model, devices=target_devices)
        return use_multi_gpus

    def forward(self, input: Any, target: Any = None, /) -> tuple[Any, torch.Tensor | None]:
        """
        Forward pass function

        - Parameters:
            - x_train: The training data
        - Returns: `Any` kind of model output
        """
        # forward model
        y = self.model(input)

        # forward loss
        if self.loss_fn is not None and target is not None:
            try:
                loss = self.loss_fn(y, target)
            except Exception as loss_error:
                runtime_error = errors.LossError()
                raise loss_error from runtime_error
        else:
            loss = None
        return y, loss

    @classmethod
    def from_checkpoint(cls, ckpt: checkpoint.Checkpoint[Any] | str, experiment: str | None = None, *, map_location: torch.device | None = None):
        """
        Method to load a manager from a saved `Checkpoint`. The manager will not be compiled with a loss function and its metrics.

        - Parameters:
            - ckpt: Either a `Checkpoint` of `Any` object or a `str` of checkpoint path
            - experiment: A `str` of the experiment name for the checkpoint
            - map_location: An optional `torch.device` to load the checkpoint
        - Returns: A loaded `Manager`
        """
        # load checkpoint
        if not isinstance(ckpt, checkpoint.Checkpoint) and experiment is not None:
            experiment = f"{experiment}.exp" if experiment.endswith(".exp") else experiment
            ckpt = checkpoint.load(experiment, ckpt)
        elif not isinstance(ckpt, checkpoint.Checkpoint):
            ckpt = checkpoint.Checkpoint.from_saved(ckpt, map_location=map_location)

        # recover model to manager
        if isinstance(ckpt.model, torch.nn.Module):
            manager = cls(cast(Module, ckpt.model), ckpt.optimizer, loss_fn=cast(Loss | None, ckpt.loss_fn), metrics=cast(dict[str, Metric], ckpt.metrics))
        elif isinstance(ckpt.model, BaseManager):
            manager = ckpt.model
            if isinstance(manager.model, torch.nn.parallel.DataParallel):
                manager.model = manager.model.module
            if manager.loss_fn is not None and hasattr(manager.loss_fn, "_metric_fn"):
                if isinstance(manager.loss_fn._metric_fn, torch.nn.parallel.DataParallel):
                    assert isinstance(manager.loss_fn._metric_fn.module, Loss), errors._raise(TypeError("Loss function is not a valid `Loss`."))
                    manager.loss_fn = manager.loss_fn._metric_fn.module
            else:
                manager.loss_fn = None
        else:
            raise TypeError(f"The saved checkpoint contains a model with type of {type(ckpt.model)} that cannot be recoverred to a `Manager`.")
        
        # initialize manager version
        if not hasattr(manager, "version"):
            manager.version = Version("v1.0")

        # convert to current version
        manager.convert()

        # reset if map location is given
        if map_location is not None:
            manager.reset(map_location)
        else:
            manager.reset()
        return manager

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
        # load state dict elements
        assert "model" in state_dict, errors._raise(KeyError("The given dictionary does not have 'model' element."))
        assert "optimizer" in state_dict, errors._raise(KeyError("The given dictionary does not have 'optimizer' element."))
        assert "loss_fn" in state_dict, errors._raise(KeyError("The given dictionary does not have 'loss_fn' element."))
        assert "metrics" in state_dict, errors._raise(KeyError("The given dictionary does not have 'metrics' element."))
        model: OrderedDict[str, Any] = state_dict["model"]
        optimizer: dict[str, Any] | None = state_dict["optimizer"]
        loss_fn: OrderedDict[str, Any] | None = state_dict["loss_fn"]
        metrics: dict[str, OrderedDict[str, Any]] = state_dict["metrics"]

        # load state dict to current model, optimizer, loss_fn, and metrics
        self.model.load_state_dict(model, *args, **kwargs)
        if optimizer is not None:
            assert self.optimizer is not None, errors._raise(ValueError("The manager has not been compiled with 'optimizer' given."))
            self.optimizer.load_state_dict(optimizer)
        if loss_fn is not None:
            assert self.loss_fn is not None, errors._raise(ValueError("The manager has not been compiled with 'loss_fn' given."))
            self.loss_fn.load_state_dict(state_dict=loss_fn, *args, **kwargs)
        for k, m in metrics.items():
            assert k in self.metric_fns, errors._raise(KeyError(f"The manager does not have a metric named '{k}'."))
            self.metric_fns[k].load_state_dict(state_dict=m, *args, **kwargs)

    def reset(self, cpu: torch.device = devices.CPU) -> None:
        """
        Reset model and loss functions, move all to CPU and empty device cache.

        - Parameters:
            - cpu: The CPU in `torch.device`
        """
        self.model = self.raw_model.to(cpu)
        self.loss_fn = self.raw_loss_fn.to(cpu) if self.raw_loss_fn is not None else self.raw_loss_fn
        devices.empty_cache()
        self.__exit__()

    def reset_metrics(self) -> None:
        """Reset all metrics, including loss function to initial state"""
        # reset loss
        if self.loss_fn is not None:
            self.loss_fn.reset()

        # reset metrics
        for m in self.metric_fns.values():
            m.reset()

    def state_dict(self, prefix: str = '', keep_vars: bool = False) -> OrderedDict[str, Any]:
        return OrderedDict({
            "model": self.model.state_dict(prefix=prefix, keep_vars=keep_vars),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "loss_fn": self.loss_fn.state_dict(keep_vars=keep_vars) if self.loss_fn is not None else None,
            "metrics": {k: m.state_dict(keep_vars=keep_vars) for k, m in self.metric_fns.items()}
        })

    def to(self, device: torch.device) -> None:
        """
        Move the elements in the manager to a target device

        - Parameters:
            - device: A target `torch.device`
        """
        # Move to device
        self.model = self.model.to(device)
        if self.loss_fn is not None:
            self.loss_fn = self.loss_fn.to(device)
        for k, m in self.metric_fns.items():
            self.metric_fns[k] = m.to(device)

    def to_checkpoint(self) -> checkpoint.Checkpoint[Self]:
        """
        Convert the current manager to a checkpoint

        - Returns: A `Checkpoint` with its model as the current manager
        """
        ckpt = checkpoint.Checkpoint(self)
        return ckpt

    def unpack_data(self, data: Collection[Any]) -> tuple[Any, Any]:
        """
        Unpacks data to input and target

        - Parameters:
            - data: A `Collection` of `Any` kind of data objects
        - Returns: A `tuple` of `Any` kind of input and `Any` kind of target
        """
        if len(data) >= 2:
            x, y, *_ = data
            return (x, y) if not isinstance(data, dict) else (data[x], data[y])
        else:
            return NotImplemented
