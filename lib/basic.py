from torchmanager_core import torch, _raise, VERSION as CURRENT_VERSION
from torchmanager_core.typing import Any, Collection, Dict, Generic, Module, Optional, OrderedDict, Self, Tuple, Union

from .losses import Loss, MultiLosses, MultiOutputsLosses, ParallelLoss
from .metrics import Metric
from .train import Checkpoint


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
    loss_fn: Optional[Union[Loss, ParallelLoss]]
    metric_fns: Dict[str, Metric]
    model: Module
    optimizer: Optional[torch.optim.Optimizer]
    version: str

    @property
    def compiled(self) -> bool:
        return True if self.loss_fn is not None and self.optimizer is not None else False

    @property
    def raw_loss_fn(self) -> Optional[Loss]:
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
        return self.model.module if isinstance(self.model, torch.nn.parallel.DataParallel) else self.model

    def __init__(self, model: Module, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[Union[Loss, Dict[str, Loss]]] = None, metrics: Dict[str, Metric] = {}) -> None:
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

        # compile
        self._compile(optimizer, loss_fn, metrics)

    def _compile(self, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[Union[Loss, Dict[str, Loss]]] = None, metrics: Dict[str, Metric] = {}) -> None:
        """
        Compiles the manager

        - Parameters:
            - loss_fn: An optional `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: An optional `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
            - optimizer: An optional `torch.optim.Optimizer` to train the model
        """
        # initialize loss
        if isinstance(loss_fn, MultiOutputsLosses) and len(loss_fn.losses) > 1:
            loss_fn_mapping: Dict[str, Loss] = {f"{name}_loss": fn for name, fn in loss_fn.losses.items()}  # type: ignore
            self.metric_fns.update(loss_fn_mapping)
        elif isinstance(loss_fn, dict):
            loss_fn_mapping: Dict[str, Loss] = {f"{name}_loss": fn for name, fn in loss_fn.items()}
            self.metric_fns.update(loss_fn_mapping)
            loss_fn = MultiLosses([l for l in loss_fn_mapping.values()])
        self.loss_fn = loss_fn

        # initialize metrics
        for name, fn in metrics.items():
            self.metric_fns[name] = fn

        # initialize optimizer
        self.optimizer = optimizer

    def compile(self, optimizer: torch.optim.Optimizer, loss_fn: Union[Loss, Dict[str, Loss]], metrics: Dict[str, Metric] = {}) -> None:
        """
        Compiles the manager

        - Parameters:
            - loss_fn: A `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: A `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
            - optimizer: A `torch.optim.Optimizer` to train the model
        """
        self._compile(optimizer, loss_fn, metrics)

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
                    assert isinstance(manager.loss_fn._metric_fn.module, Loss), _raise(TypeError("Loss function is not a valid `Loss`."))
                    manager.loss_fn = manager.loss_fn._metric_fn.module
            else:
                manager.loss_fn = None
        else:
            raise TypeError(f"The saved checkpoint contains a model with type of {type(ckpt.model)} that cannot be recoverred to a `Manager`.")
        return manager

    def load_state_dict(self, state_dict: OrderedDict[str, Any], strict: bool = True) -> None:
        # load state dict elements
        assert "model" in state_dict, _raise(KeyError("The given dictionary does not have 'model' element."))
        assert "optimizer" in state_dict, _raise(KeyError("The given dictionary does not have 'optimizer' element."))
        assert "loss_fn" in state_dict, _raise(KeyError("The given dictionary does not have 'loss_fn' element."))
        assert "metrics" in state_dict, _raise(KeyError("The given dictionary does not have 'metrics' element."))
        model: OrderedDict[str, Any] = state_dict["model"]
        optimizer: Optional[Dict[str, Any]] = state_dict["optimizer"]
        loss_fn: Optional[OrderedDict[str, Any]] = state_dict["loss_fn"]
        metrics: Dict[str, OrderedDict[str, Any]] = state_dict["metrics"]

        # load state dict to current model, optimizer, loss_fn, and metrics
        self.model.load_state_dict(model, strict=strict)  # type: ignore
        if optimizer is not None:
            assert self.optimizer is not None, _raise(ValueError("The manager has not been compiled with 'optimizer' given."))
            self.optimizer.load_state_dict(optimizer)
        if loss_fn is not None:
            assert self.loss_fn is not None, _raise(ValueError("The manager has not been compiled with 'loss_fn' given."))
            self.loss_fn.load_state_dict(loss_fn)
        assert metrics is not None, _raise(ValueError("The given dictionary must have 'metrics' element not to be None."))
        for k, m in metrics.items():
            assert k in self.metric_fns, _raise(KeyError(f"The manager does not have a metric named '{k}'."))
            self.metric_fns[k].load_state_dict(m)

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

    def unpack_data(self, data: Collection) -> Tuple[Any, Any]:
        """
        Unpacks data to input and target

        - Parameters:
            - data: `Any` kind of data object
        - Returns: A `tuple` of `Any` kind of input and `Any` kind of target
        """
        if len(data) >= 2:
            return tuple(data)
        else:
            return NotImplemented
