from torchmanager.callbacks import Callback as _Callback # type: ignore
from torchmanager.core.typing import Callable, Dict, Optional

class _Training(_Callback):
    """A Protected Training callback"""
    on_batch_end_fn: Optional[Callable[[int, Dict[str, float]], None]]
    on_epoch_end_fn: Optional[Callable[[int, Dict[str, float], Optional[Dict[str, float]]], None]]

    def __init__(self, on_batch_end: Optional[Callable[[int, Dict[str, float]], None]] = None, on_epoch_end: Optional[Callable[[int, Dict[str, float], Optional[Dict[str, float]]], None]] = None) -> None:
        super().__init__()
        self.on_batch_end_fn = on_batch_end
        self.on_epoch_end_fn = on_epoch_end

    def on_batch_end(self, batch: int, summary: Dict[str, float] = ...) -> None:
        if self.on_batch_end_fn is not None:
            self.on_batch_end_fn(batch, summary)
        else: pass

    def on_epoch_end(self, epoch: int, summary: Dict[str, float] = ..., val_summary: Optional[Dict[str, float]] = None) -> None:
        if self.on_epoch_end_fn is not None:
            self.on_epoch_end_fn(epoch, summary, val_summary)
        else: pass

def on_batch_end(fn: Callable[[int, Dict[str, float]], None]) -> _Training:
    """Decorator wrapper function to wrap a `Callable` function into a `_Training` callback to run the function at the end of each batch"""
    return _Training(on_batch_end=fn)

def on_epoch_end(fn: Callable[[int, Dict[str, float], Optional[Dict[str, float]]], None]) -> _Training:
    """Decorator wrapper function to wrap a `Callable` function into a `_Training` callback to run the function at the end of each epoch"""
    return _Training(on_epoch_end=fn)