from typing import Callable
from torchmanager_core import abc
from torchmanager_core.typing import Any, Generic, Optional, SupportsFloat, TypeVar

from .callback import FrequencyCallback
from .protocols import Frequency, SummaryWriteble, Weighted

W = TypeVar('W', bound=Weighted)
Writer = TypeVar("Writer", bound=SummaryWriteble)

class DynamicWeight(FrequencyCallback, abc.ABC, Generic[W]):
    '''
    An abstract dynamic weight callback that set weight dynamically

    * extends: `.callback.Callback`
    * abstract class that needs implementation of `step` method
    '''
    __weighted: W
    __writer: Optional[SummaryWriteble]

    @property
    def _weighted(self) -> W:
        return self.__weighted

    @property
    def _writer(self) -> Optional[SummaryWriteble]:
        return self.__writer

    def __init__(self, weighted: W, freq: Frequency = Frequency.EPOCH, writer: Optional[Writer] = None) -> None:
        '''
        Constructor

        - Parameters:
            - weighted: A targeted object that performs `Weighted` protocol
            - freq: A `WeightUpdateFreq` of the frequency type to update the weight
            - writer: An optional writer that performs `SummaryWritable` protocol
        '''
        super().__init__(freq=freq)
        self.__weighted = weighted
        self.__writer = writer
        self.current_step = 0

    def _update(self, result: Any) -> None:
        self._weighted.weight = result

    def on_epoch_end(self, epoch: int, *args: Any, **kwargs: Any) -> None:
        # update
        super().on_epoch_end(epoch, *args, **kwargs)
        
        # write results to Tensorboard
        if self._writer is not None and isinstance(self._weighted.weight, SupportsFloat):
            # get summary
            w = self._weighted.weight
            key = f"{type(self._weighted)}.weight"
            result = {key: w}
            self._writer.add_scalars("train", result, epoch)

class LambdaDynamicWeight(DynamicWeight[W], Generic[W]):
    '''
    A dynamic weight callback that set weight dynamically with lambda function
    
    * extends: `DynamicWeight`
    '''
    __lambda_fn: Callable[[int], Any]

    def __init__(self, fn: Callable[[int], Any], weighted: W, freq: Frequency = Frequency.EPOCH, writer: Optional[Writer] = None) -> None:
        super().__init__(weighted, freq, writer)
        self.__lambda_fn = fn

    def step(self, *args: Any, **kwargs: Any) -> Any:
        return self.__lambda_fn(self.current_step)