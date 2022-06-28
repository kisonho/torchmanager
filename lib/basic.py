from __future__ import annotations
from typing import Iterable

from torchmanager_core import torch, view
from torchmanager_core.typing import Any, Callable, Dict, Generic, Module, Optional, SizedIterable, Tuple, Type, Union

from .losses import Loss, MultiLosses, MultiOutputsLosses
from .metrics import Metric
from .train import Checkpoint

class BaseManager(Generic[Module]):
    """
    The basic manager

    - Properties:
        - compiled: A `bool` flag of if the manager has been compiled
        - loss_fn: A `Callable` method that takes the truth and predictions in `torch.Tensor` and returns a loss `torch.Tensor`
        - metrics: A `dict` of metrics with a name in `str` and a `Callable` method that takes the truth and predictions in `torch.Tensor` and returns a loss `torch.Tensor`
        - model: A target `torch.nn.Module` to be trained
        - optimizer: A `torch.optim.Optimizer` to train the model
    """
    # properties
    loss_fn: Optional[Loss]
    metric_fns: Dict[str, Metric]
    model: Module
    optimizer: Optional[torch.optim.Optimizer]

    @property
    def compiled(self) -> bool:
        return True if self.loss_fn is not None and self.optimizer is not None else False
    
    def __init__(self, model: Module, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[Union[Loss, Dict[str, Loss], Callable[[Any, Any], torch.Tensor]]] = None, metrics: Dict[str, Union[Metric, Callable[[Any, Any], torch.Tensor]]] = {}) -> None:
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

        # compile
        self._compile(optimizer, loss_fn, metrics)

    def _compile(self, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[Union[Loss, Dict[str, Loss], Callable[[Any, Any], torch.Tensor]]] = None, metrics: Dict[str, Union[Metric, Callable[[Any, Any], torch.Tensor]]] = {}) -> None:
        """
        Compiles the manager
        
        - Parameters:
            - loss_fn: An optional `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: An optional `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
            - optimizer: An optional `torch.optim.Optimizer` to train the model
        """
        # initialize loss
        if isinstance(loss_fn, MultiOutputsLosses) and len(loss_fn.losses) > 1:
            loss_fn_mapping: Dict[str, Loss] = {f"{name}_loss": fn for name, fn in loss_fn.losses.items()} # type: ignore
            self.metric_fns.update(loss_fn_mapping)
        elif isinstance(loss_fn, dict):
            loss_fn_mapping: Dict[str, Loss] = {f"{name}_loss": fn for name, fn in loss_fn.items()}
            self.metric_fns.update(loss_fn_mapping)
            loss_fn = MultiLosses([l for l in loss_fn_mapping.values()])
        elif loss_fn is not None and not isinstance(loss_fn, Loss):
            loss_fn = Loss(loss_fn)
            view.warnings.warn("[Deprecation Warning]: parsing `loss_fn` as a function was deprecated from v1.0.0 and will no longer be available from v1.1.0, use losses.Loss object instead.", DeprecationWarning)
        self.loss_fn = loss_fn

        # initialize metrics
        for name, fn in metrics.items():
            assert "loss" not in name, "[Manager Error]: Keyward 'loss' should not be contained in metrics."
            if isinstance(fn, Metric): self.metric_fns[name] = fn
            else:
                view.warnings.warn("[Deprecation Warning]: parsing a metric in `metrics` as a function was deprecated from v1.0.0 and will no longer be available from v1.1.0, use `metrics.Metric` object instead.", DeprecationWarning)
                self.metric_fns[name] = Metric(fn)

        # initialize optimizer
        self.optimizer = optimizer

    def compile(self, optimizer: torch.optim.Optimizer, loss_fn: Union[Loss, Dict[str, Loss], Callable[[Any, Any], torch.Tensor]], metrics: Dict[str, Union[Metric, Callable[[Any, Any], torch.Tensor]]] = {}) -> None:
        """
        Compiles the manager
        
        - Parameters:
            - loss_fn: A `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: A `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
            - optimizer: A `torch.optim.Optimizer` to train the model
        """
        self._compile(optimizer, loss_fn, metrics)

    @classmethod
    def from_checkpoint(cls: Type[BaseManager[torch.nn.Module]], ckpt: Checkpoint) -> BaseManager[torch.nn.Module]:
        """
        Method to load a manager from a saved `Checkpoint`. The manager will not be compiled with a loss function and its metrics.

        - Returns: A loaded `Manager`
        """
        # recover model to manager
        if isinstance(ckpt.model, torch.nn.Module):
            manager = cls(ckpt.model, ckpt.optimizer, loss_fn=ckpt.loss_fn, metrics=ckpt.metrics)
            return manager
        elif isinstance(ckpt.model, BaseManager):
            return ckpt.model
        else: raise TypeError(f"[Ckpt Error]: The saved checkpoint contains a model with type of {type(ckpt.model)} that cannot be recoverred to a `Manager`.")

    def to_checkpoint(self) -> Checkpoint[Module]:
        """
        Convert the current manager to a checkpoint
        
        - Returns: A `Checkpoint` with its model in `Module` type
        """
        metrics: Dict[str, torch.nn.Module] = {k: m for k, m in self.metric_fns.items()}
        ckpt = Checkpoint(self.model, optimizer=self.optimizer, loss_fn=self.loss_fn, metrics=metrics)
        return ckpt

class DataManager:
    def unpack_data(self, data: SizedIterable) -> Tuple[Any, Any]:
        """
        Unpacks data to input and target
        
        - Parameters:
            - data: `Any` kind of data object
        - Returns: A `tuple` of `Any` kind of input and `Any` kind of target
        """
        if len(data) == 2:
            return tuple(data)
        else: return NotImplemented