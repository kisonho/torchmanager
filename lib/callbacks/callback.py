from __future__ import annotations
from ..core.typing import Dict, Optional

class Callback:
    """
    A training callback
    """
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
