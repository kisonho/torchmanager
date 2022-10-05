from torchmanager_core import sys
from torchmanager_core.typing import Dict, List, Optional

from .ckpt import Callback, MonitorType

class StopTraining(RuntimeError):
    '''
    A runtime error to stop training
    
    - Properties:
        - epoch: An `int` of the epoch index when training stopped
    '''
    epoch: int

    def __init__(self, epoch: int, *args: object) -> None:
        super().__init__(*args)
        self.epoch = epoch

class EarlyStop(Callback):
    '''
    The early stop callback that raises `StopTraining` error during the training if monitored metric not improved for several steps

    - Properties:
        - monitor: A `str` of monitored metric
        - monitor_type: A `MonitorType` of either `MIN` of `MAX` mode for the best model
        - steps: An `int` of steps to monitor
    '''
    __metrics: List[float]
    monitor: str
    monitor_type: MonitorType
    steps: int

    def __init__(self, monitor: str, monitor_type: MonitorType = MonitorType.MAX, steps: int = 10) -> None:
        super().__init__()
        self.__metrics = [sys.float_info.min if monitor_type == MonitorType.MAX else sys.float_info.max]
        self.monitor = monitor
        self.monitor_type = monitor_type
        self.steps = steps

    def on_epoch_end(self, epoch: int, summary: Dict[str, float] = ..., val_summary: Optional[Dict[str, float]] = None) -> None:
        # load monitoring value
        if val_summary is not None: summary = val_summary
        monitoring_value = summary[self.monitor]

        # compare with recorded metrics
        max_value = max(self.__metrics)
        if monitoring_value < max_value and len(self.__metrics) >= self.steps: raise StopTraining(epoch)
        elif len(self.__metrics) >= self.steps: self.__metrics.pop()
        self.__metrics.insert(0, monitoring_value)