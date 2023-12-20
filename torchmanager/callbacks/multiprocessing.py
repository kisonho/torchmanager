from torchmanager_core import torch, _raise
from torchmanager_core.multiprocessing import Process
from torchmanager_core.typing import Any, Callable, Iterator, Optional

from .callback import Callback


class MultiCallbacks(Callback):
    """
    A callback that contains multiple callbacks

    * extends: `Callbacks`

    - Properties:
        - callbacks_list: A `list` of callbacks in `Callback`
        - num_workers: An optional `int` of the number of workers to use in multiprocessing
    """
    callbacks_list: list[Callback]
    num_workers: Optional[int]

    def __init__(self, *callbacks: Callback, num_workers: Optional[int] = None) -> None:
        """
        Constructor

        - Parameters:
            - callbacks: A `list` of callbacks in `Callback`
            - num_workers: An optional `int` of the number of workers to use in multiprocessing, a `None` or 1 to disable multiprocessing
        """
        super().__init__()
        self.callbacks_list = list(callbacks)
        self.num_workers = num_workers
        assert self.num_workers is None or self.num_workers > 0, _raise(ValueError("The num_workers must be greater than 0 if given."))

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
        """
        Check if processes have exceptions and raise them

        - Parameters:
            - processes: A `list` of processes in `Process`
        """
        # loop through processes
        for process in processes:
            # check if process has exception
            if process.exception is not None:
                # raise exception
                raise process.exception

    def on_batch_end(self, batch: int, summary: dict[str, float] = {}) -> None:
        callback_fns: list[Callable[..., None]] = [callback.on_batch_end for callback in self.callbacks_list]
        self.run(callback_fns, batch, summary)

    def on_batch_start(self, batch: int) -> None:
        callback_fns: list[Callable[..., None]] = [callback.on_batch_start for callback in self.callbacks_list]
        self.run(callback_fns, batch)

    def on_epoch_end(self, epoch: int, summary: dict[str, float] = {}, val_summary: Optional[dict[str, float]] = None) -> None:
        callback_fns: list[Callable[..., None]] = [callback.on_epoch_end for callback in self.callbacks_list]
        self.run(callback_fns, epoch, summary, val_summary)

    def on_epoch_start(self, epoch: int) -> None:
        callback_fns: list[Callable[..., None]] = [callback.on_epoch_start for callback in self.callbacks_list]
        self.run(callback_fns, epoch)

    def on_train_end(self, model: torch.nn.Module) -> None:
        callback_fns: list[Callable[..., None]] = [callback.on_train_end for callback in self.callbacks_list]
        self.run(callback_fns, model)

    def on_train_start(self, initial_epoch: int = 0) -> None:
        callback_fns: list[Callable[..., None]] = [callback.on_train_start for callback in self.callbacks_list]
        self.run(callback_fns, initial_epoch)

    def pop(self, index: int = -1) -> Callback:
        """
        Remove and return the callback at the index

        - Parameters:
            - index: The index of the callback to remove in `int`
        - Returns: The callback removed in `Callback`
        """
        return self.callbacks_list.pop(index)

    def run(self, funcs: list[Callable[..., None]], *args: Any, **kwargs: Any) -> None:
        """
        Run a function 

        - Parameters:
            - func: The function to run in the process
            - args: The positional arguments to pass to the function
            - kwargs: The keyword arguments to pass to the function
        """
        # check if multiprocessing is enabled
        if self.num_workers is None or self.num_workers <= 1:
            # loop through functions
            for func in funcs:
                # run function
                func(*args, **kwargs)
            return

        # create processes
        processes = [Process(func, *args, **kwargs) for func in funcs]

        # start processes
        for process in processes:
            process.run()

        # join processes
        for process in processes:
            process.join()

        # check exceptions
        self.check_processes_exceptions(processes)
