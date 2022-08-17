from torchmanager_core import abc
from torchmanager_core.typing import Any, Dict, Optional

from .protocols import Frequency

class Callback:
    """An empty basic training callback"""
    def on_batch_end(self, batch: int, summary: Dict[str, float]={}) -> None:
        """
        The callback when batch ends

        - Parameters:
            - batch: An `int` of batch index
            - summary: A `dict` of summary with name in `str` and value in `float`
        """
        pass

    def on_batch_start(self, batch: int) -> None:
        """
        The callback when batch starts

        - Parameters:
            - batch: An `int` of batch index
        """
        pass

    def on_epoch_end(self, epoch: int, summary: Dict[str, float]={}, val_summary: Optional[Dict[str, float]]=None) -> None:
        """
        The callback when batch ends

        - Parameters:
            - epoch: An `int` of epoch index
            - summary: A `dict` of training summary with name in `str` and value in `float`
            - val_summary: A `dict` of validation summary with name in `str` and value in `float`
        """
        pass

    def on_epoch_start(self, epoch: int) -> None:
        """
        The callback when epoch starts

        - Parameters:
            - epoch: An `int` of epoch index
        """
        pass

    def on_train_end(self) -> None:
        """The callback when training ends"""
        pass

    def on_train_start(self, initial_epoch: int = 0) -> None:
        """
        The callback when training starts
        
        - Parameters:
            - initial_epoch: An `int` of initial epoch index
        """
        pass

class FrequencyCallback(Callback, abc.ABC):
    '''
    A callback with frequency control

    * extends: `Callbacks`
    * abstract class that needs implementation of `_update` and `step` method

    - Parameters:
        - current_step: An `int` of the current step index
        - freq: A `WeightUpdateFreq` of the frequency type to update the weight
    '''
    __step: int
    freq: Frequency
    '''The frequency of this callback'''

    @property
    def current_step(self) -> int:
        '''The current step index'''
        return self.__step

    @current_step.setter
    def current_step(self, step: int) -> None:
        assert step >= 0, "The step index must be a non-negative number."
        self.__step = step

    def __init__(self, freq: Frequency = Frequency.EPOCH) -> None:
        super().__init__()
        self.__step = 0
        self.freq = freq

    @abc.abstractmethod
    def _update(self, result: Any) -> None: pass

    def on_batch_end(self, batch: int, summary: Dict[str, float] = {}) -> None:
        if self.freq == Frequency.BATCH:
            result = self.step()
            self._update(result)
            self.current_step += 1

    def on_epoch_end(self, epoch: int, summary: Dict[str, float] = {}, val_summary: Optional[Dict[str, float]] = None) -> None:
        if self.freq == Frequency.EPOCH:
            result = self.step(summary, val_summary)
            self._update(result)
            self.current_step += 1

    @abc.abstractmethod
    def step(self, summary: Dict[str, float] = {}, val_summary: Optional[Dict[str, float]] = None) -> Any:
        '''
        Abstract method to step the callback
        
        - Returns: An `Any` type of result for the new step
        '''
        return NotImplemented