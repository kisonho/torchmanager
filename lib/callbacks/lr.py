from torchmanager_core import torch
from torchmanager_core.typing import Any, Dict, Optional, Protocol, runtime_checkable

from ..train import learning_rate
from .callback import Callback

@runtime_checkable
class _SummaryWriter(Protocol):
    """The SummaryWriter protocol"""
    def add_scalars(self, main_tag: str, tag_scalar_dict: Any, global_step: Optional[int] = None) -> None: pass

class LrSchedueler(Callback):
    """
    The callback to step learning rate scheduler

    - Parameters:
        - freq: An `_lr.LrScheduleFreq` of the frequency to update learning rate
    """
    __lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    _writer: Optional[_SummaryWriter]
    freq: learning_rate.LrScheduleFreq

    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler, freq: learning_rate.LrScheduleFreq = learning_rate.LrScheduleFreq.EPOCH, tf_board_writer: Optional[_SummaryWriter] = None) -> None:
        super().__init__()
        self.__lr_scheduler = scheduler
        assert isinstance(tf_board_writer, _SummaryWriter) or tf_board_writer is None, "[Callback Error]: The given writer does not performs to SummaryWriter protocol."
        self._writer = tf_board_writer
        self.freq = freq

    def on_batch_end(self, *args: Any, **kwargs: Any) -> None:
        if self.freq == learning_rate.LrScheduleFreq.BATCH: self.__lr_scheduler.step()

    def on_epoch_end(self, epoch: int, *args: Any, **kwargs: Any) -> None:
        # update lr scheduler
        if self.freq == learning_rate.LrScheduleFreq.EPOCH: self.__lr_scheduler.step()
        
        # write results to Tensorboard
        if self._writer is not None:
            # get summary
            summary = {}
            lr_list = self.__lr_scheduler.get_last_lr()
            if len(lr_list) > 1:
                for i, lr in enumerate(lr_list):
                    summary[f'lr_{i}'] = lr
            else: summary['lr'] = lr_list[0]

            # record summary
            for key in summary.keys():
                result: Dict[str, float] = {}
                result["train"] = summary[key]
                self._writer.add_scalars(key, result, epoch + 1)

    def on_train_start(self, initial_epoch: int = 0) -> None:
        learning_rate.initial_step_lr_scheduler(self.__lr_scheduler, initial_epoch)
