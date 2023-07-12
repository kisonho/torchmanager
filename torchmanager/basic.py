from torchmanager_core import devices, errors, torch, Version, deprecated, API_VERSION, VERSION as CURRENT_VERSION
from torchmanager_core.checkpoint import Checkpoint
from torchmanager_core.protocols import Resulting
from torchmanager_core.typing import Any, Collection, Generic, Module, Optional, OrderedDict, Self, Union

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
        - loss_fn: An optional `Loss` for the objective function
        - metrics: A `dict` of metrics with a name in `str` as keys and a `Metric` as values
        - model: A target `torch.nn.Module` to be trained
        - optimizer: A `torch.optim.Optimizer` to train the model
        - raw_loss_fn: An optional `Loss` of the non-paralleled loss function
        - raw_model: A non-paralleled target `torch.nn.Module` model
    """
    # properties
    loss_fn: Optional[Resulting]
    """The optional main loss function in `Resulting`"""
    metric_fns: dict[str, Resulting]
    """A `dict` of the metric functions with names as keys in `str` and metric functions as values in `torch.metrics.Metric`"""
    model: Union[Module, torch.nn.DataParallel]
    optimizer: Optional[torch.optim.Optimizer]
    version: Version

    @property
    def compiled(self) -> bool:
        """The `bool` flag of if this manager has been compiled"""
        return True if self.loss_fn is not None and self.optimizer is not None else False

    @property
    def raw_loss_fn(self) -> Optional[Resulting]:
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
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model  # type: ignore

    def __init__(self, model: Module, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[Union[Loss, dict[str, Loss]]] = None, metrics: dict[str, Metric] = {}) -> None:
        """
        Constructor

        - Parameters:
            - loss_fn: An optional `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: An optional `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
            - model: An optional target `torch.nn.Module` to be trained
            - optimizer: An optional `torch.optim.Optimizer` to train the model
        """
        # initialize
        self.metric_fns = {}
        self.model = model
        self.version = CURRENT_VERSION

        # initialize loss
        if isinstance(loss_fn, dict):
            loss_fn_mapping: dict[str, Loss] = {f"{name}_loss": fn for name, fn in loss_fn.items()}
            self.metric_fns.update(loss_fn_mapping)
            loss_fn = MultiLosses([l for l in loss_fn_mapping.values()])
        self.loss_fn = loss_fn

        # initialize metrics
        for name, fn in metrics.items():
            self.metric_fns[name] = fn

        # initialize optimizer
        self.optimizer = optimizer

    @deprecated('v1.3', 'v1.4')
    def _compile(self, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[Union[Loss, dict[str, Loss]]] = None, metrics: dict[str, Metric] = {}) -> None:
        """
        Compiles the manager

        - Parameters:
            - loss_fn: An optional `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: An optional `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
            - optimizer: An optional `torch.optim.Optimizer` to train the model
        """
        # initialize loss
        if isinstance(loss_fn, dict):
            loss_fn_mapping: dict[str, Loss] = {f"{name}_loss": fn for name, fn in loss_fn.items()}
            self.metric_fns.update(loss_fn_mapping)
            loss_fn = MultiLosses([l for l in loss_fn_mapping.values()])
        self.loss_fn = loss_fn

        # initialize metrics
        for name, fn in metrics.items():
            self.metric_fns[name] = fn

        # initialize optimizer
        self.optimizer = optimizer

    @deprecated('v1.3', 'v1.4')
    def compile(self, optimizer: torch.optim.Optimizer, loss_fn: Union[Loss, dict[str, Loss]], metrics: dict[str, Metric] = {}) -> None:
        """
        Recompiles the manager with optimizer loss function and metrics

        - Parameters:
            - loss_fn: A `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: A `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
            - optimizer: A `torch.optim.Optimizer` to train the model
        """
        self._compile(optimizer, loss_fn, metrics)

    def convert(self, from_version: Version) -> None:
        """
        Convert from a version to current version

        - Parameters:
            - from_version: A `torchmanager.version.Version` to convert from
        """
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
        assert isinstance(self.raw_loss_fn, Loss), errors._raise(TypeError("The loss function is not a valid `Loss` object."))
        paralleled_loss_fn, use_multi_gpus = devices.data_parallel(self.raw_loss_fn, devices=target_devices, parallel_type=ParallelLoss)
        assert isinstance(paralleled_loss_fn, ParallelLoss) or isinstance(paralleled_loss_fn, Loss), errors._raise(TypeError("Paralleled function is not a valid `ParallelLoss` or `Loss` after parallel."))
        self.loss_fn = paralleled_loss_fn

        # multi gpus support for model
        self.model, use_multi_gpus = devices.data_parallel(self.raw_model, devices=target_devices)
        return use_multi_gpus

    @classmethod
    def from_checkpoint(cls, ckpt: Union[Checkpoint[Any], str], map_location: Optional[torch.device] = None):
        """
        Method to load a manager from a saved `Checkpoint`. The manager will not be compiled with a loss function and its metrics.

        - Parameters:
            - ckpt: Either a `Checkpoint` of `Any` object or a `str` of checkpoint path
            - map_location: An optional `torch.device` to load the checkpoint
        - Returns: A loaded `Manager`
        """
        # load checkpoint
        if not isinstance(ckpt, Checkpoint):
            ckpt = Checkpoint.from_saved(ckpt, map_location=map_location)

        # recover model to manager
        if isinstance(ckpt.model, torch.nn.Module):
            manager = cls(ckpt.model, ckpt.optimizer, loss_fn=ckpt.loss_fn, metrics=ckpt.metrics)  # type: ignore
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
        manager.convert(manager.version)
        return manager

    def load_state_dict(self, state_dict: OrderedDict[str, Any], strict: bool = True) -> None:
        # load state dict elements
        assert "model" in state_dict, errors._raise(KeyError("The given dictionary does not have 'model' element."))
        assert "optimizer" in state_dict, errors._raise(KeyError("The given dictionary does not have 'optimizer' element."))
        assert "loss_fn" in state_dict, errors._raise(KeyError("The given dictionary does not have 'loss_fn' element."))
        assert "metrics" in state_dict, errors._raise(KeyError("The given dictionary does not have 'metrics' element."))
        model: OrderedDict[str, Any] = state_dict["model"]
        optimizer: Optional[dict[str, Any]] = state_dict["optimizer"]
        loss_fn: Optional[OrderedDict[str, Any]] = state_dict["loss_fn"]
        metrics: dict[str, OrderedDict[str, Any]] = state_dict["metrics"]

        # load state dict to current model, optimizer, loss_fn, and metrics
        self.model.load_state_dict(model, strict=strict)  # type: ignore
        if optimizer is not None:
            assert self.optimizer is not None, errors._raise(ValueError("The manager has not been compiled with 'optimizer' given."))
            self.optimizer.load_state_dict(optimizer)
        if loss_fn is not None:
            assert self.loss_fn is not None, errors._raise(ValueError("The manager has not been compiled with 'loss_fn' given."))
            self.loss_fn.load_state_dict(state_dict=loss_fn)
        assert metrics is not None, errors._raise(ValueError("The given dictionary must have 'metrics' element not to be None."))
        for k, m in metrics.items():
            assert k in self.metric_fns, errors._raise(KeyError(f"The manager does not have a metric named '{k}'."))
            self.metric_fns[k].load_state_dict(state_dict=m)

    def reset(self, cpu: torch.device = devices.CPU) -> None:
        """
        Reset model and loss functions, move all to CPU and empty device cache

        - Parameters:
            - cpu: The CPU in `torch.device`
        """
        self.model = self.raw_model.to(cpu)
        self.loss_fn = self.raw_loss_fn.to(cpu) if self.raw_loss_fn is not None else self.raw_loss_fn
        devices.empty_cache()

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

    def to_checkpoint(self) -> Checkpoint[Self]:
        """
        Convert the current manager to a checkpoint

        - Returns: A `Checkpoint` with its model as the current manager
        """
        ckpt = Checkpoint(self)
        return ckpt

    def unpack_data(self, data: Collection[Any]) -> tuple[Any, Any]:
        """
        Unpacks data to input and target

        - Parameters:
            - data: A `Collection` of `Any` kind of data objects
        - Returns: A `tuple` of `Any` kind of input and `Any` kind of target
        """
        if len(data) >= 2:
            return tuple(data)
        else:
            return NotImplemented
