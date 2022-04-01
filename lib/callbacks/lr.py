from torch.utils import tensorboard
from ..core import torch
from ..core._typing import Dict, Optional
from ..train import _lr
from .callback import Callback

class LrSchedueler(Callback):
    """
    The callback to step learning rate scheduler

    - Parameters:
        - freq: An `_lr.LrScheduleFreq` of the frequency to update learning rate
    """
    __lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    _writer: Optional[tensorboard.writer.SummaryWriter]
    freq: _lr.LrScheduleFreq

    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler, freq: _lr.LrScheduleFreq, tf_board_writer: Optional[tensorboard.writer.SummaryWriter] = None) -> None:
        super().__init__()
        self.__lr_scheduler = scheduler
        self._writer = tf_board_writer
        self.freq = freq

    def on_batch_end(self, *args, **kwargs) -> None:
        if self.freq == _lr.LrScheduleFreq.BATCH:
            self.__lr_scheduler.step()

    def on_epoch_end(self, epoch: int, *args, **kwargs) -> None:
        # update lr scheduler
        if self.freq == _lr.LrScheduleFreq.EPOCH:
            self.__lr_scheduler.step()

        # get summary
        summary = {}
        lr_list = self.__lr_scheduler.get_last_lr()
        if len(lr_list) > 1:
            for i, lr in enumerate(lr_list):
                summary[f'lr_{i}'] = lr
        else: summary['lr'] = lr_list[0]
        
        # write results to Tensorboard
        if self._writer is not None:
            for key in summary.keys():
                result: Dict[str, float] = {}
                result["train"] = summary[key]
                self._writer.add_scalars(key, result, epoch + 1)
