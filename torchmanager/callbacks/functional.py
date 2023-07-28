from torchmanager_core.protocols import Frequency
from torchmanager_core.typing import Any, Callable, Optional

from .callback import FrequencyCallback


class LambdaCallback(FrequencyCallback):
    """
    The callback for lambda functions

    * extends: `.callback.FrequencyCallback`

    - Properties:
        - pre_callback_fn: An optional function that accepts the current step as `int` for callbacks before a batch or epoch starts
        - post_callback_fn: An optional function that accepts the current step as `int` for callbacks after a batch or epoch starts
    """
    pre_callback_fn: Optional[Callable[[int], None]]
    post_callback_fn: Optional[Callable[[int, dict[str, Any]], None]]

    def __init__(self, pre_fn: Optional[Callable[[int], None]] = None, post_fn: Optional[Callable[[int, dict[str, Any]], None]] = None, *, freq: Frequency = Frequency.EPOCH, initial_step: int = 0) -> None:
        super().__init__(freq, initial_step)
        self.pre_callback_fn = pre_fn
        self.post_callback_fn = post_fn

    def step(self, summary: dict[str, float] = {}, val_summary: Optional[dict[str, float]] = None) -> None:
        # combine summary
        if val_summary is not None:
            summary.update({f"val_{k}": v for k, v in val_summary.items()})

        # call functions
        if self.pre_callback_fn is not None and (self.freq == Frequency.BATCH_START or self.freq == Frequency.EPOCH_START):
            self.pre_callback_fn(self.current_step)
        elif self.post_callback_fn is not None and (self.freq == Frequency.BATCH or self.freq == Frequency.EPOCH):
            self.post_callback_fn(self.current_step, summary)


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


def on_batch_start(fn: Callable[[int], None]) -> LambdaCallback:
    """
    The wrapper function to wrap a function as a callback with `on_batch_start` called

    * Usage:
    >>> @on_batch_start
    >>> def callback_fn(batch_index: int) -> None:
    ...     ...

    >>> callbacks_list = [..., callback_fn]
    """
    return LambdaCallback(pre_fn=fn, freq=Frequency.BATCH_START)


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


def on_epoch_start(fn: Callable[[int], None]) -> LambdaCallback:
    """
    The wrapper function to wrap a function as a callback with `on_epoch_start` called

    * Usage:
    >>> @on_epoch_start
    >>> def callback_fn(epoch_index: int) -> None:
    ...     ...

    >>> callbacks_list = [..., callback_fn]
    """
    return LambdaCallback(pre_fn=fn, freq=Frequency.EPOCH_START)
