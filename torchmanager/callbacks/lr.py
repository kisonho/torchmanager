from torchmanager_core.protocols import Frequency, SummaryWriteble
from torchmanager_core.typing import Any, Generic, TypeVar

from .callback import FrequencyCallback

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
_LrScheduler = LRScheduler


Scheduler = TypeVar("Scheduler", bound=_LrScheduler)


class LrSchedueler(FrequencyCallback, Generic[Scheduler]):
    """
    The callback to step learning rate scheduler

    * extends: `FrequencyCallback`
    * implements: `torchmanager_core.protocols.LrStepping`

    - Parameters:
        - freq: An `..train.learning_rate.LrScheduleFreq` of the frequency to update learning rate
    """
    __lr_scheduler: Scheduler
    __name: str
    __writer: SummaryWriteble | None

    @property
    def _name(self) -> str:
        return self.__name

    @property
    def _scheduler(self) -> Scheduler:
        return self.__lr_scheduler

    @property
    def _writer(self) -> SummaryWriteble | None:
        return self.__writer

    def __init__(self, scheduler: Scheduler, freq: Frequency = Frequency.EPOCH, name: str = 'lr', tf_board_writer: SummaryWriteble | None = None) -> None:
        super().__init__(freq)
        self.__lr_scheduler = scheduler
        self.__name = name
        self.__writer = tf_board_writer

    def _update(self, result: Any) -> None:
        pass

    def on_epoch_end(self, epoch: int, summary: dict[str, float], val_summary: dict[str, Any] | None = None) -> None:
        # get lr summary
        lr_summary = {}
        lr_list = self._scheduler.get_last_lr()
        if len(lr_list) > 1:
            for i, lr in enumerate(lr_list):
                lr_summary[f'{self._name}_{i}'] = lr
        else:
            lr_summary[self._name] = lr_list[0]

        # write results to Tensorboard
        if self._writer is not None:
            # record summary
            for key in lr_summary.keys():
                result: dict[str, float] = {}
                result["train"] = lr_summary[key]
                self._writer.add_scalars(key, result, epoch)

        # update lr scheduler
        summary.update(lr_summary)
        super().on_epoch_end(epoch, summary, val_summary)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        self._scheduler.step()
