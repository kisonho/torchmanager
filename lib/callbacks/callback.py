from torchmanager_core import abc, torch
from torchmanager_core.protocols import Frequency
from torchmanager_core.typing import Any, Dict, Optional

class Callback:
    """An empty basic training callback"""

    def on_batch_end(self, batch: int, summary: Dict[str, float] = {}) -> None:
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

    def on_epoch_end(self, epoch: int, summary: Dict[str, float] = {}, val_summary: Optional[Dict[str, float]] = None) -> None:
        """
        The callback when batch ends

        - Parameters:
            - epoch: An `int` of epoch index
            - summary: A `dict` of training summary with name in `str` and value in `float`
            - val_summary: An optional `dict` of validation summary with name in `str` and value in `float`
        """
        pass

    def on_epoch_start(self, epoch: int) -> None:
        """
        The callback when epoch starts

        - Parameters:
            - epoch: An `int` of epoch index
        """
        pass

    def on_train_end(self, model: torch.nn.Module) -> None:
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
    """
    A callback with frequency control

    * extends: `Callbacks`
    * abstract class that needs implementation of `step` method

    - Properties:
        - current_step: An `int` of the current step index
        - freq: A `WeightUpdateFreq` of the frequency type to update the weight
    """

    __freq: Frequency
    __step: int

    @property
    def current_step(self) -> int:
        """The current step index"""
        return self.__step

    @current_step.setter
    def current_step(self, step: int) -> None:
        assert step >= 0, "The step index must be a non-negative number."
        self.__step = step

    @property
    def freq(self) -> Frequency:
        return self.__freq

    def __init__(self, freq: Frequency = Frequency.EPOCH, initial_step: int = 0) -> None:
        """
        Constructor

        - Parameters:
            - freq: A `.protocols.Frequency` of callback frequency
            - initial_step: An `int` of the initial step that starts with
        """
        super().__init__()
        self.__freq = freq
        self.__step = initial_step

    def _update(self, result: Any) -> None:
        """
        Method to update with the result after step

        - Parameters:
            - result: `Any` kind of result value
        """
        pass

    def on_batch_end(self, batch: int, summary: Dict[str, float] = {}) -> None:
        if self.freq == Frequency.BATCH:
            result = self.step(summary)
            self._update(result)
            self.current_step += 1

    def on_train_start(self, initial_epoch: int = 0) -> None:
        if self.freq == Frequency.EPOCH:
            self.current_step = initial_epoch

    def on_epoch_end(self, epoch: int, summary: Dict[str, float] = {}, val_summary: Optional[Dict[str, float]] = None) -> None:
        if self.freq == Frequency.EPOCH:
            result = self.step(summary, val_summary)
            self._update(result)
            self.current_step += 1

    @abc.abstractmethod
    def step(self, summary: Dict[str, float], val_summary: Optional[Dict[str, float]] = None) -> Any:
        """
        Abstract method to step the callback
        
        - Parameters:
            - summary: A `dict` of training summary with name in `str` and value in `float`
            - val_summary: An optional `dict` of validation summary with name in `str` and value in `float`
        - Returns: An `Any` type of result for the new step
        """
        return NotImplemented
