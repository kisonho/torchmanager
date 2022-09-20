from torchmanager_core import abc
from torchmanager_core.typing import Any, Callable, Generic, Optional, SupportsFloat, TypeVar

from .callback import FrequencyCallback
from .protocols import Frequency, SummaryWriteble, Weighted

W = TypeVar('W', bound=Weighted)

class DynamicWeight(FrequencyCallback, abc.ABC, Generic[W]):
    '''
    An abstract dynamic weight callback that set weight dynamically

    * extends: `.callback.Callback`
    * abstract class that needs implementation of `step` method
    '''
    __key: str
    __weighted: W
    __writer: Optional[SummaryWriteble]

    @property
    def _key(self) -> str:
        return self.__key

    @property
    def _weighted(self) -> W:
        return self.__weighted

    @property
    def _writer(self) -> Optional[SummaryWriteble]:
        return self.__writer

    def __init__(self, weighted: W, freq: Frequency = Frequency.EPOCH, writer: Optional[SummaryWriteble] = None, name: Optional[str] = None) -> None:
        '''
        Constructor

        - Parameters:
            - weighted: A targeted object that performs `Weighted` protocol
            - freq: A `WeightUpdateFreq` of the frequency type to update the weight
            - writer: An optional writer that performs `SummaryWritable` protocol
        '''
        super().__init__(freq=freq)
        self.__key = f"{type(weighted).__name__}.weight" if name is None else name
        self.__weighted = weighted
        self.__writer = writer
        self.current_step = 0

    def _update(self, result: Any) -> None:
        self._weighted.weight = result

    def on_epoch_end(self, epoch: int, *args: Any, **kwargs: Any) -> None:
        # write results to Tensorboard
        if self._writer is not None and isinstance(self._weighted.weight, SupportsFloat):
            # get summary
            w = self._weighted.weight
            result = {'train': w}
            self._writer.add_scalars(self._key, result, epoch)

        # update
        super().on_epoch_end(epoch, *args, **kwargs)

class LambdaDynamicWeight(DynamicWeight[W], Generic[W]):
    '''
    A dynamic weight callback that set weight dynamically with lambda function
    
    * extends: `DynamicWeight`
    '''
    _lambda_fn: Callable[[int], Any]

    def __init__(self, fn: Callable[[int], Any], weighted: W, freq: Frequency = Frequency.EPOCH, writer: Optional[SummaryWriteble] = None, name: Optional[str] = None) -> None:
        super().__init__(weighted, freq, writer, name)
        self._lambda_fn = fn

    def step(self, *args: Any, **kwargs: Any) -> Any:
        return self._lambda_fn(self.current_step)