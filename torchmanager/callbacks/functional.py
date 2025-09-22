from torchmanager_core import _raise
from torchmanager_core.protocols import Frequency
from torchmanager_core.typing import Any, Callable, TypeAlias

from .callback import Callback

PreCallbackFn: TypeAlias = Callable[[int], None]
PostCallbackFn: TypeAlias = Callable[[int, dict[str, Any]], None]


class LambdaCallback(Callback):
    """
    The callback for lambda functions

    * extends: `.callback.FrequencyCallback`

    - Properties:
        - freq: A `Frequency` of the callback frequency
        - pre_callback_fn: An optional function that accepts the current batch or epoch number as `int` for callbacks before a batch or epoch starts
        - post_callback_fn: An optional function that accepts the current batch or epoch number as `int` for callbacks after a batch or epoch starts
    """
    __freq: Frequency
    pre_callback_fn: PreCallbackFn | None
    post_callback_fn: PostCallbackFn | None

    @property
    def freq(self) -> Frequency:
        return self.__freq

    @freq.setter
    def freq(self, f: Frequency) -> None:
        # check if the frequency is valid
        assert f in [Frequency.BATCH, Frequency.EPOCH], _raise(TypeError(f"The frequency must be either `Frequency.BATCH` or `Frequency.EPOCH`, {f} is not supported."))
        self.__freq = f

    def __init__(self, pre_fn: PreCallbackFn | None = None, post_fn: PostCallbackFn | None = None, *, freq: Frequency = Frequency.EPOCH) -> None:
        super().__init__()
        self.freq = freq
        self.pre_callback_fn = pre_fn
        self.post_callback_fn = post_fn

    def on_batch_end(self, batch: int, summary: dict[str, float] = {}) -> None:
        if self.freq == Frequency.BATCH and self.post_callback_fn is not None:
            self.post_callback_fn(batch, summary)

    def on_batch_start(self, batch: int) -> None:
        if self.freq == Frequency.BATCH and self.pre_callback_fn is not None:
            self.pre_callback_fn(batch)

    def on_epoch_end(self, epoch: int, summary: dict[str, float] = {}, val_summary: dict[str, float] | None = None) -> None:
        if self.freq == Frequency.EPOCH and self.post_callback_fn is not None:
            # add validation summary if exists
            if val_summary is not None:
                val_summary = {"val_" + k: v for k, v in val_summary.items()}
                summary.update(val_summary)
            self.post_callback_fn(epoch, summary)

    def on_epoch_start(self, epoch: int) -> None:
        if self.freq == Frequency.EPOCH and self.pre_callback_fn is not None:
            self.pre_callback_fn(epoch)


def on_batch_end(fn: Callable[[int, dict[str, float]], None]) -> LambdaCallback:
    """
    The wrapper function to wrap a function as a callback with `on_batch_end` called

    * Usage:
    >>> @on_batch_end
    >>> def callback_fn(batch_index: int, batch_summary: dict[str, float]) -> None:
    ...     ...

    >>> callbacks_list = [..., callback_fn]
    """
    return LambdaCallback(post_fn=fn, freq=Frequency.BATCH)


def on_batch_start(fn: PreCallbackFn) -> LambdaCallback:
    """
    The wrapper function to wrap a function as a callback with `on_batch_start` called

    * Usage:
    >>> @on_batch_start
    >>> def callback_fn(batch_index: int) -> None:
    ...     ...

    >>> callbacks_list = [..., callback_fn]
    """
    return LambdaCallback(pre_fn=fn, freq=Frequency.BATCH)


def on_epoch_end(fn: Callable[[int, dict[str, float]], None]) -> LambdaCallback:
    """
    The wrapper function to wrap a function as a callback with `on_epoch_end` called

    * Usage:
    >>> @on_epoch_end
    >>> def callback_fn(epoch_index: int, epoch_summary: dict[str, float]) -> None:
    ...     ...

    >>> callbacks_list = [..., callback_fn]
    """
    return LambdaCallback(post_fn=fn, freq=Frequency.EPOCH)


def on_epoch_start(fn: PreCallbackFn) -> LambdaCallback:
    """
    The wrapper function to wrap a function as a callback with `on_epoch_start` called

    * Usage:
    >>> @on_epoch_start
    >>> def callback_fn(epoch_index: int) -> None:
    ...     ...

    >>> callbacks_list = [..., callback_fn]
    """
    return LambdaCallback(pre_fn=fn, freq=Frequency.EPOCH)
