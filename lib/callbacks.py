# import typing modules
from typing import Dict, Optional
from enum import Enum

# import required modules
import sys, torch
from torch.utils import tensorboard

class Callback:
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
        return

    def on_batch_start(self, batch: int) -> None:
        '''
        The callback when batch starts

        - Parameters:
            - batch: An `int` of batch index
        '''
        return

    def on_epoch_end(self, epoch: int, summary: Dict[str, float]={}, val_summary: Optional[Dict[str, float]]=None) -> None:
        '''
        The callback when batch ends

        - Parameters:
            - epoch: An `int` of epoch index
            - summary: A `dict` of training summary with name in `str` and value in `float`
            - val_summary: A `dict` of validation summary with name in `str` and value in `float`
        '''
        return

    def on_epoch_start(self, epoch: int) -> None:
        '''
        The callback when epoch starts

        - Parameters:
            - epoch: An `int` of epoch index
        '''
        return

    def on_train_end(self) -> None:
        '''The callback when training ends'''
        return

    def on_train_start(self) -> None:
        '''The callback when training starts'''
        return

class Checkpoint(Callback):
    '''
    The callback to save the latest checkpoint for each epoch

    - Properties:
        - model: A `torch.nn.Module` to be saved
        - ckpt_path: A `str` of checkpoint directory
    '''
    # properties
    __model: torch.nn.Module
    __path: str

    @property
    def model(self) -> torch.nn.Module:
        return self.__model

    @property
    def ckpt_path(self) -> str:
        return self.__path

    def __init__(self, model: torch.nn.Module, ckpt_path: str) -> None:
        super().__init__()
        self.__model = model
        self.__path = ckpt_path

    def on_epoch_end(self, *args, **kwargs) -> None:
        torch.save(self.model, self.ckpt_path)

class MonitorType(Enum):
    '''The enum of monitor types'''
    MIN: int=0
    MAX: int=1

    @property
    def init_score(self) -> float:
        if self == MonitorType.MAX:
            return -1
        elif self == MonitorType.MIN:
            sys.float_info.max
        else:
            raise ValueError(f'[MonitorType Error]: Monitor type {self} is not supported.')

class BestCheckpoint(Checkpoint):
    '''
    The callback to save the latest checkpoint for each epoch

    - Properties:
        - best_score: A `float` of the best score to be monitored
    '''
    # properties
    __monitor: str
    __monitor_type = MonitorType.MAX
    best_score: float

    @property
    def monitor(self) -> str:
        return self.__monitor

    @property
    def monitor_type(self) -> MonitorType:
        return self.__monitor_type

    def __init__(self, monitor: str, *args, monitor_type: MonitorType=MonitorType.MAX, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__monitor = monitor
        self.__monitor_type = monitor_type
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
    _writer: tensorboard.SummaryWriter

    @property
    def writer(self) -> tensorboard.SummaryWriter:
        return self._writer

    def __init__(self, log_dir: str) -> None:
        '''
        Constructor

        - Parameters:
            - log_dir: A `str` of logging directory
        '''
        super().__init__()
        self._writer = tensorboard.SummaryWriter(log_dir)

    def on_epoch_end(self, epoch: int, summary: Dict[str, float]={}, val_summary: Optional[Dict[str, float]]=None) -> None:
        # write results to Tensorboard
        for key in summary.keys():
            result: Dict[str, float] = {}
            result["train"] = summary[key]
            if val_summary is not None:
                result["val"] = val_summary[key]
            self.writer.add_scalars(key, result, epoch + 1)