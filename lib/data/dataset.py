from torch.utils.data import Dataset as _Dataset, DataLoader, IterableDataset
from torchmanager_core import abc, devices, math, os, torch
from torchmanager_core.typing import Any, Callable, Iterator, Sequence, Tuple, TypeVar
from torchmanager_core.view import warnings

T = TypeVar("T")


class Dataset(IterableDataset[T], abc.ABC):
    """
    A dataset that iterates with batch size

    * extends: `IterableDataset`
    * implements: `typing.Collection`
    * Abstract class
    * Used as a combination of `torch.utils.data.IterableDataset` and `torch.utils.data.DataLoader`

    >>> from torchmanager import Manager
    >>> class SomeDataset(Dataset):
    ...    @property
    ...    def unbatched_size(self) -> int: ...
    ...
    ...    def __init__(self, ...,  batch_size: int, device: torch.device = devices.CPU) -> None: ...
    ...    def __getitem__(self, index: Any) -> Any: ...
    >>> dataset = SomeDataset(..., batch_size)
    >>> manager = Manager(...)
    >>> manager.fit(dataset, ...)

    - Properties:
        - batch_size: An `int` of batch size for the current dataset
        - device: A `torch.device` for the data to be pinned during iteration
        - drop_last: A `bool` flag of if drop the last data that not enought for the batch size
        - shuffle: A `bool` flag of if shuffling the data
    """

    __batch_size: int
    __device: torch.device
    drop_last: bool
    shuffle: bool

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, b: int) -> None:
        if b <= 0:
            raise ValueError("Batch size must be a positive number")
        self.__batch_size = b

    @property
    def device(self) -> torch.device:
        _, device, target_devices = devices.search(self.__device)
        devices.set_default(target_devices[0])
        return device

    @property
    @abc.abstractmethod
    def unbatched_len(self) -> int:
        return NotImplemented

    def __init__(self, batch_size: int, device: torch.device = devices.CPU, drop_last: bool = False, shuffle: bool = False) -> None:
        """
        Constructor

        - Parameters:
            - batch_size: An `int` of batch size for the current dataset
            - device: A `torch.device` for the data to be pinned during iteration
            - drop_last: A `bool` flag of if drop the last data that not enought for the batch size
            - shuffle: A `bool` flag of if shuffling the data
        """
        super().__init__()
        self.__device = device
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __contains__(self, value: Any) -> bool:
        for i in range(len(self)):
            if self[i] == value:
                return True
        return False

    @abc.abstractmethod
    def __getitem__(self, index: Any) -> Any:
        """Returns an unbatched item"""
        return NotImplemented

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        # initialize devices
        cpu_count = os.cpu_count()
        if cpu_count is None:
            cpu_count = 0
        device = self.device

        # yield data
        data_loader = DataLoader(self, batch_size=self.batch_size, drop_last=self.drop_last, shuffle=self.shuffle, num_workers=cpu_count, pin_memory=(device == devices.CPU))
        for data in data_loader:
            yield self.unpack_data(data)

    def __len__(self) -> int:
        if self.drop_last:
            return int(self.unbatched_len / self.batch_size)
        else:
            return math.ceil(self.unbatched_len / self.batch_size)

    @staticmethod
    def unpack_data(data: Any) -> Tuple[Any, Any]:
        """
        Unpacks a single data into inputs and targets

        - Parameters:
            - data: `Any` kind of single data
        - Returns: A `tuple` of `Any` kind of inputs and `Any` kind of targets
        """
        if isinstance(data, Sequence):
            return data[0], data[1] if len(data) >= 2 else NotImplemented
        else:
            return NotImplemented


def batched(fn: Callable[..., _Dataset]):
    """
    Wrap a loading PyTorch dataset function into a loading dataset function

    Use as decorator with a function:
    >>> from torch.utils.data import Dataset as TorchDataset
    >>> from torchmanager_core import devices
    >>> @batched
    >>> def load_some_dataset(...) -> TorchDataset:
    ...     ...
    >>> some_dataset: DataLoader = load_some_dataset(..., batch_size=4, device=devices.GPU, drop_last=True, shuffle=True)

    Or with a class:
    >>> @batched
    >>> class SomeDataset(TorchDataset):
    ...     ...
    >>> some_dataset: DataLoader = SomeDataset(..., batch_size=4, device=devices.GPU, drop_last=True, shuffle=True)

    - Parameters in the wrapped function:
        - batch_size: An `int` of batch size for the current dataset
        - device: A `torch.device` for the data to be pinned during iteration
        - drop_last: A `bool` flag of if drop the last data that not enought for the batch size
        - shuffle: A `bool` flag of if shuffling the data
    - Returns: A wrapped function that accepts a loading function which returns `torch.utils.data.Dataset` and returns a loading function which returns `DataLoader`
    """
    # wrap function
    def wrapped_fn(*args: Any, batch_size: int = 1, device: torch.device = devices.CPU, drop_last: bool = False, shuffle: bool = False, **kwargs: Any) -> DataLoader:
        # initialize devices
        cpu_count = os.cpu_count()
        if cpu_count is None:
            cpu_count = 0
        _, device, targeted_devices = devices.search(device)
        if device == devices.CPU:
            pin_memory = False
        else:
            devices.set_default(targeted_devices[0])
            pin_memory = True

        # load dataset
        loaded_dataset = fn(*args, **kwargs)
        if isinstance(loaded_dataset, Dataset):
            warnings.warn("The loaded dataset is a `torchmanager.data.Dataset` which has already been wrapped with batch loader during iteration.", RuntimeWarning)
        data_loader = DataLoader(loaded_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory, num_workers=cpu_count)
        return data_loader
    return wrapped_fn
