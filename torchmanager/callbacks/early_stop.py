from torchmanager_core import errors, sys
from torchmanager_core.typing import Optional

from .ckpt import Callback, MonitorType


class EarlyStop(Callback):
    '''
    The early stop callback that raises `StopTraining` error during the training if monitored metric not improved for several steps

    - Properties:
        - monitor: A `str` of monitored metric
        - monitor_type: A `MonitorType` of either `MIN` of `MAX` mode for the best model
        - steps: An `int` of steps to monitor
    '''
    __metrics: list[float]
    monitor: str
    monitor_type: MonitorType
    steps: int

    @property
    def _metrics(self) -> list[float]:
        return self.__metrics

    def __init__(self, monitor: str, monitor_type: MonitorType = MonitorType.MAX, steps: int = 10) -> None:
        super().__init__()
        self.__metrics = [sys.float_info.min if monitor_type == MonitorType.MAX else sys.float_info.max]
        self.monitor = monitor
        self.monitor_type = monitor_type
        self.steps = steps

    def on_epoch_end(self, epoch: int, summary: dict[str, float] = ..., val_summary: Optional[dict[str, float]] = None) -> None:
        # load monitoring value
        summary = val_summary if val_summary is not None else summary
        monitoring_value = summary[self.monitor]

        # compare with recorded metrics
        value = self._metrics[0]
        if self.monitor_type == MonitorType.MAX and monitoring_value <= value and len(self._metrics) > self.steps:
            raise errors.StopTraining(epoch)
        elif self.monitor_type == MonitorType.MAX and monitoring_value > value:
            self._metrics.clear()
        elif self.monitor_type == MonitorType.MIN and monitoring_value >= value and len(self._metrics) > self.steps:
            raise errors.StopTraining(epoch)
        elif self.monitor_type == MonitorType.MIN and monitoring_value < value:
            self._metrics.clear()
        self._metrics.append(monitoring_value)
