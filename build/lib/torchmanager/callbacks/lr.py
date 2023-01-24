from torchmanager_core import torch
from torchmanager_core.protocols import Frequency, SummaryWriteble
from torchmanager_core.typing import Any, Dict, Generic, Optional, TypeVar

from .callback import FrequencyCallback

Scheduler = TypeVar("Scheduler", bound=torch.optim.lr_scheduler._LRScheduler)
Writer = TypeVar("Writer", bound=SummaryWriteble)

class LrSchedueler(FrequencyCallback, Generic[Scheduler]):
    """
    The callback to step learning rate scheduler

    * extends: `FrequencyCallback`

    - Parameters:
        - freq: An `..train.learning_rate.LrScheduleFreq` of the frequency to update learning rate
    """
    __lr_scheduler: Scheduler
    __name: str
    __writer: Optional[SummaryWriteble]

    @property
    def _name(self) -> str:
        return self.__name

    @property
    def _scheduler(self) -> Scheduler:
        return self.__lr_scheduler

    @property
    def _writer(self) -> Optional[SummaryWriteble]:
        return self.__writer

    def __init__(self, scheduler: Scheduler, freq: Frequency = Frequency.EPOCH, name: str = 'lr', tf_board_writer: Optional[Writer] = None) -> None:
        super().__init__(freq)
        self.__lr_scheduler = scheduler
        self.__name = name
        self.__writer = tf_board_writer

    def _update(self, result: Any) -> None:
        pass

    def on_epoch_end(self, epoch: int, summary: Dict[str, float], val_summary: Optional[Dict[str, Any]] = None) -> None:
        # get lr summary
        lr_summary = {}
        lr_list = self._scheduler.get_last_lr()
        if len(lr_list) > 1:
            for i, lr in enumerate(lr_list):
                lr_summary[f'{self._name}_{i}'] = lr
        else: lr_summary[self._name] = lr_list[0]

        # write results to Tensorboard
        if self._writer is not None:
            # record summary
            for key in lr_summary.keys():
                result: Dict[str, float] = {}
                result["train"] = lr_summary[key]
                self._writer.add_scalars(key, result, epoch)

        # update lr scheduler
        summary.update(lr_summary)
        super().on_epoch_end(epoch, summary, val_summary)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        self._scheduler.step()