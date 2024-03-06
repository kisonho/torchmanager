from torchmanager_core import torch, view, _raise
from torchmanager_core.typing import Optional

from .callback import Callback


class ProgressBar(Callback):
    """
    A callback to display a progress bar.

    * extends: `Callback`

    - Properties:
        - progress_bar: A `tqdm` progress bar
    """
    iterations_per_epoch: int
    progress_bar: Optional[view.tqdm]
    verbose_type: view.VerboseType

    def __init__(self, iterations_per_epoch: int, *, verbose_type: view.VerboseType = view.VerboseType.ALL) -> None:
        """
        Constructor

        - Parameters:
            - iterations_per_epoch: The number of iterations per epoch in `int`
        """
        super().__init__()
        self.iterations_per_epoch = iterations_per_epoch
        self.progress_bar = None
        self.verbose_type = verbose_type

    def on_epoch_start(self, epoch: int) -> None:
        # create progress bar
        self.progress_bar = view.tqdm(total=self.iterations_per_epoch)

    def on_epoch_end(self, epoch: int, summary: dict[str, float] = {}, val_summary: Optional[dict[str, float]] = None) -> None:
        # close progress bar
        assert self.progress_bar is not None, _raise(TypeError("Progress bar is not initialized."))
        self.progress_bar.close()
        self.progress_bar = None

    def on_batch_end(self, batch: int, summary: dict[str, float] = {}) -> None:# initialize progress summary
        # switch verbose type
        if self.verbose_type == view.VerboseType.LOSS:
            progress_summary = {name: s for name, s in summary.items() if "loss" in name}
        elif self.verbose_type == view.VerboseType.METRICS:
            progress_summary = {name: s for name, s in summary.items() if "loss" not in name}
        elif self.verbose_type == view.VerboseType.ALL:
            progress_summary = summary
        else:
            raise TypeError(f"Verbose type {self.verbose_type} is not supported.")

        # update progress bar
        assert self.progress_bar is not None, _raise(TypeError("Progress bar is not initialized."))
        self.progress_bar.set_postfix(progress_summary)
        self.progress_bar.update()

    def on_train_end(self, model: torch.nn.Module) -> None:
        # close progress bar
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None
