from torchmanager_core import torch
from torchmanager_core.multiprocessing import Process
from torchmanager_core.typing import Iterator, Optional

from .callback import Callback


class MultiCallbacks(Callback):
    """
    A callback that contains multiple callbacks and run with multiprocessing

    * extends: `Callbacks`

    - Properties:
        - callbacks_list: A `list` of callbacks in `Callback`
    """
    callbacks_list: list[Callback]

    def __init__(self, *callbacks: Callback) -> None:
        """
        Constructor

        - Parameters:
            - callbacks: A `list` of callbacks in `Callback`
        """
        super().__init__()
        self.callbacks_list = list(callbacks)

    def __getitem__(self, index: int) -> Callback:
        return self.callbacks_list[index]

    def __iter__(self) -> Iterator[Callback]:
        return iter(self.callbacks_list)

    def __len__(self) -> int:
        return len(self.callbacks_list)

    def append(self, callback: Callback) -> None:
        """
        Append a callback to the list

        - Parameters:
            - callback: The callback to append in `Callback`
        """
        self.callbacks_list.append(callback)

    def check_processes_exceptions(self, processes: list[Process]) -> None:
        # loop through processes
        for process in processes:
            # check if process has exception
            if process.exception is not None:
                # raise exception
                raise process.exception

    def on_batch_end(self, batch: int, summary: dict[str, float] = {}) -> None:
        # create processes
        processes = [Process(callback.on_batch_end, batch, summary) for callback in self.callbacks_list]

        # start processes
        for process in processes:
            process.run()

        # join processes
        for process in processes:
            process.join()

        # check exceptions
        self.check_processes_exceptions(processes)

    def on_batch_start(self, batch: int) -> None:
        # create processes
        processes = [Process(callback.on_batch_start, batch) for callback in self.callbacks_list]

        # start processes
        for process in processes:
            process.run()

        # join processes
        for process in processes:
            process.join()

        # check exceptions
        self.check_processes_exceptions(processes)

    def on_epoch_end(self, epoch: int, summary: dict[str, float] = {}, val_summary: Optional[dict[str, float]] = None) -> None:
        # create processes
        processes = [Process(callback.on_epoch_end, epoch, summary, val_summary) for callback in self.callbacks_list]

        # start processes
        for process in processes:
            process.run()

        # join processes
        for process in processes:
            process.join()

        # check exceptions
        self.check_processes_exceptions(processes)

    def on_epoch_start(self, epoch: int) -> None:
        # create processes
        processes = [Process(callback.on_epoch_start, epoch) for callback in self.callbacks_list]

        # start processes
        for process in processes:
            process.run()

        # join processes
        for process in processes:
            process.join()

        # check exceptions
        self.check_processes_exceptions(processes)

    def on_train_end(self, model: torch.nn.Module) -> None:
        # create processes
        processes = [Process(callback.on_train_end, model) for callback in self.callbacks_list]

        # start processes
        for process in processes:
            process.run()

        # join processes
        for process in processes:
            process.join()

        # check exceptions
        self.check_processes_exceptions(processes)

    def on_train_start(self, initial_epoch: int = 0) -> None:
        # create processes
        processes = [Process(callback.on_train_start, initial_epoch) for callback in self.callbacks_list]

        # start processes
        for process in processes:
            process.run()

        # join processes
        for process in processes:
            process.join()

        # check exceptions
        self.check_processes_exceptions(processes)

    def pop(self, index: int = -1) -> Callback:
        """
        Remove and return the callback at the index

        - Parameters:
            - index: The index of the callback to remove in `int`
        - Returns: The callback removed in `Callback`
        """
        return self.callbacks_list.pop(index)
