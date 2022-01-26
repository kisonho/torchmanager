# import typing modules
from __future__ import annotations
from typing import Any, Dict, Optional, OrderedDict, Type
from enum import Enum

# import required modules
import sys, torch
from torch.utils.tensorboard.writer import SummaryWriter

# import core modules
from .losses import Loss
from .metrics import Metric

class Callback():
    '''
    A training callback
    '''
    def on_batch_end(self, batch: int, summary: Dict[str, float]={}) -> None:
        '''
        The callback when batch ends

        - Parameters:
            - batch: An `int` of batch index
            - summary: A `dict` of summary with name in `str` and value in `float`
        '''
        pass

    def on_batch_start(self, batch: int) -> None:
        '''
        The callback when batch starts

        - Parameters:
            - batch: An `int` of batch index
        '''
        pass

    def on_epoch_end(self, epoch: int, summary: Dict[str, float]={}, val_summary: Optional[Dict[str, float]]=None) -> None:
        '''
        The callback when batch ends

        - Parameters:
            - epoch: An `int` of epoch index
            - summary: A `dict` of training summary with name in `str` and value in `float`
            - val_summary: A `dict` of validation summary with name in `str` and value in `float`
        '''
        pass

    def on_epoch_start(self, epoch: int) -> None:
        '''
        The callback when epoch starts

        - Parameters:
            - epoch: An `int` of epoch index
        '''
        pass

    def on_train_end(self) -> None:
        '''The callback when training ends'''
        pass

    def on_train_start(self) -> None:
        '''The callback when training starts'''
        pass

class Checkpoint(Callback):
    '''
    The callback to save the latest checkpoint for each epoch

    - Properties:
        - ckpt_path: A `str` of checkpoint directory
        - model: A `torch.nn.Module` to be saved
        - optimizer: A `torch.nn.Optimizer` to be saved
        - save_weights_only: A `bool` flag of if only save state_dict of model
    '''
    # properties
    ckpt_path: str
    last_epoch: int = 0
    loss_fn: Optional[Loss] = None
    metrics: Optional[dict[str, Metric]] = None
    model: torch.nn.Module
    optimizer: Optional[torch.optim.Optimizer] = None
    save_weights_only: bool = False

    def __init__(self, model: torch.nn.Module, ckpt_path: str, epoch: int=0, optimizer: Optional[torch.optim.Optimizer]=None, loss_fn: Optional[Loss]=None, metrics: Optional[dict[str, Metric]]=None, save_weights_only: bool=False) -> None:
        '''
        Constructor

        - Parameters:
            - model: A target `torch.nn.Module`
            - ckpt_path: A `str` of file path
            - epoch: An `int` of epoch index
            - optimizer: An optional `torch.optim.Optimizer` to be recorded
            - loss_fn: An optional `Loss` to be recorded
            - metrics: An optional `dict` of the metrics with key in `str` and value in `Metric` to be recorded
        '''
        super().__init__()
        self.last_epoch = epoch
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.model = model
        self.optimizer = optimizer
        self.ckpt_path = ckpt_path
        self.save_weights_only = save_weights_only

    @classmethod
    def from_saved(cls: Type[Checkpoint], ckpt_path: str, model: Optional[torch.nn.Module]=None) -> Checkpoint:
        '''
        Load checkpoint from a saved checkpoint file

        - Parameters:
            - ckpt_path: A `str` of file path
            - model: An optional `torch.nn.Module` for structure when only weights is saved
        '''
        # load checkpint dictionary
        ckpt: Dict[str, Any] = torch.load(ckpt_path)

        # load model
        if ckpt["save_weights_only"] is True:
            assert model is not None, "[Checkpoint Error]: The structure model is not given."
            state_dict: OrderedDict[str, torch.Tensor] = ckpt["model"]
            model.load_state_dict(state_dict)
            ckpt["model"] = model
        else:
            if model is not None:
                m: torch.nn.Module = ckpt["model"]
                model.load_state_dict(m.state_dict())
                ckpt["model"] = m
        return cls(**ckpt)

    def on_epoch_end(self, epoch: int, *args, **kwargs) -> None:
        self.last_epoch = epoch
        ckpt = self.__dict__
        if self.save_weights_only is True:
            ckpt["model"] = self.model.state_dict()
        torch.save(ckpt, self.ckpt_path)

class MonitorType(Enum):
    '''The enum of monitor types'''
    MIN = int(0)
    MAX = int(1)

    @property
    def init_score(self) -> float:
        if self == MonitorType.MAX:
            return -1
        elif self == MonitorType.MIN:
            return sys.float_info.max
        else:
            raise ValueError(f'[MonitorType Error]: Monitor type {self} is not supported.')

class BestCheckpoint(Checkpoint):
    '''
    The callback to save the latest checkpoint for each epoch

    - Properties:
        - best_score: A `float` of the best score to be monitored
        - monitor: A `str` of the summary name to be monitored
        - monitor_type: A `MonitorType` of the monitor
    '''
    # properties
    best_score: float
    monitor: str
    monitor_type = MonitorType.MAX

    def __init__(self, monitor: str, *args, monitor_type: MonitorType=MonitorType.MAX, **kwargs) -> None:
        '''
        Constructor

        - Parameters:
            - monitor: A `str` of monitored metric
            - monitor_type: A `MonitorType` of either `MIN` of `MAX` mode for the best model
        '''
        super().__init__(*args, **kwargs)
        self.monitor = monitor
        self.monitor_type = monitor_type
        self.best_score = monitor_type.init_score

    def on_epoch_end(self, *args, summary: Dict[str, float]={}, val_summary: Optional[Dict[str, float]]=None, **kwargs) -> None:
        # get score
        score = val_summary[self.monitor] if val_summary is not None else summary[self.monitor]

        # save when best
        if score >= self.best_score and self.monitor_type == MonitorType.MAX:
            self.best_score = score
            super().on_epoch_end(*args, **kwargs)
        elif score <= self.best_score and self.monitor_type == MonitorType.MIN:
            self.best_score = score
            super().on_epoch_end(*args, **kwargs)

class TensorBoard(Callback):
    '''
    The callback to record summary to tensorboard for each epoch

    - Properties:
        - writer: A `tensorboard.SummaryWriter` to record scalars
    '''
    # properties
    _writer: SummaryWriter

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    def __init__(self, log_dir: str) -> None:
        '''
        Constructor

        - Parameters:
            - log_dir: A `str` of logging directory
        '''
        super().__init__()
        self._writer = SummaryWriter(log_dir)

    def on_epoch_end(self, epoch: int, summary: Dict[str, float]={}, val_summary: Optional[Dict[str, float]]=None) -> None:
        # write results to Tensorboard
        for key in summary.keys():
            result: Dict[str, float] = {}
            result["train"] = summary[key]
            if val_summary is not None:
                result["val"] = val_summary[key]
            self.writer.add_scalars(key, result, epoch + 1)