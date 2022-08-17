from torchmanager_core import torch
from torchmanager_core.typing import Any, Dict, Optional, TypeVar

from ..train import learning_rate
from .callback import FrequencyCallback
from .protocols import Frequency, SummaryWriteble

Writer = TypeVar("Writer", bound=SummaryWriteble)

class LrSchedueler(FrequencyCallback):
    """
    The callback to step learning rate scheduler

    * extends: `FrequencyCallback`

    - Parameters:
        - freq: An `..train.learning_rate.LrScheduleFreq` of the frequency to update learning rate
    """
    __lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    __writer: Optional[SummaryWriteble]

    @property
    def _scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return self.__lr_scheduler

    @property
    def _writer(self) -> Optional[SummaryWriteble]:
        return self.__writer

    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler, freq: Frequency = Frequency.EPOCH, tf_board_writer: Optional[Writer] = None) -> None:
        super().__init__(freq)
        self.__lr_scheduler = scheduler
        self.__writer = tf_board_writer

    def _update(self, result: Any) -> None:
        pass

    def on_epoch_end(self, epoch: int, *args: Any, **kwargs: Any) -> None:
        # update lr scheduler
        super().on_epoch_end(epoch, *args, **kwargs)
        
        # write results to Tensorboard
        if self.__writer is not None:
            # get summary
            summary = {}
            lr_list = self._scheduler.get_last_lr()
            if len(lr_list) > 1:
                for i, lr in enumerate(lr_list):
                    summary[f'lr_{i}'] = lr
            else: summary['lr'] = lr_list[0]

            # record summary
            for key in summary.keys():
                result: Dict[str, float] = {}
                result["train"] = summary[key]
                self.__writer.add_scalars(key, result, epoch + 1)

    def on_train_start(self, initial_epoch: int = 0) -> None:
        learning_rate.initial_step_lr_scheduler(self._scheduler, initial_epoch)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        self._scheduler.step()
