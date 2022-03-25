from __future__ import annotations
from ..core import os, sys, tensorboard, torch, view
from ..core._typing import Dict, Enum, Optional, Tuple
from ..train.checkpoint import Checkpoint as Ckpt

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

class LastCheckpoint(Callback):
    """
    The callback to save the last checkpoint during training

    - Properties:
        - ckpt_path: A `str` of checkpoint path
    """
    _checkpoint: Ckpt
    ckpt_path: str

    def __init__(self, model: torch.nn.Module, ckpt_path: str, **kwargs) -> None:
        '''
        Constructor

        - Parameters:
            - model: A target `torch.nn.Module`
            - ckpt_path: A `str` of the checkpoint path
            - **kwargs: Other arguments in `Checkpoint` constructor
        '''
        super().__init__()
        self._checkpoint = Ckpt(model, **kwargs)
        self.ckpt_path = os.path.normpath(ckpt_path)

    def on_epoch_end(self, epoch: int, *args, **kwargs) -> None:
        self._checkpoint.save(epoch, self.ckpt_path)

class Checkpoint(LastCheckpoint):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        view.warnings.warn("[Deprecation Warning]: Checkpoint callback has been renamed to LastCheckpoint and was deprecated from v 1.0.0, it will be removed at v1.1.0.", DeprecationWarning)

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

class BestCheckpoint(LastCheckpoint):
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
    monitor_type: MonitorType

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
    _writer: tensorboard.writer.SummaryWriter

    @property
    def writer(self) -> tensorboard.writer.SummaryWriter:
        return self._writer

    def __init__(self, log_dir: str) -> None:
        '''
        Constructor

        - Parameters:
            - log_dir: A `str` of logging directory
        '''
        super().__init__()
        self._writer = tensorboard.writer.SummaryWriter(log_dir)

    def add_graph(self, model: torch.nn.Module, input_shape: Optional[Tuple[int, ...]] = None) -> None:
        '''
        Add graph to TensorBoard

        - Parameters:
            - model: A `torch.nn.Module` to add
            - input_shape: An optional `tuple` of in `int` for the inputs
        '''
        inputs = torch.randn(input_shape) if input_shape is not None else None
        self._writer.add_graph(model, input_to_model=inputs)

    def on_epoch_end(self, epoch: int, summary: Dict[str, float]={}, val_summary: Optional[Dict[str, float]]=None) -> None:
        # write results to Tensorboard
        for key in summary.keys():
            result: Dict[str, float] = {}
            result["train"] = summary[key]
            if val_summary is not None and key in val_summary:
                result["val"] = val_summary[key]
            self.writer.add_scalars(key, result, epoch + 1)