from torch.utils.data import Dataset as _Dataset, DataLoader
from torchmanager_core import abc, devices, math, os, torch, _raise
from torchmanager_core.typing import Any, Callable, Iterator, Sequence, TypeVar

T = TypeVar("T")


class Dataset(_Dataset[T], abc.ABC):
    """
    A dataset that iterates with batch size

    * extends: `torch.utils.data.Dataset`
    * implements: `typing.Collection`
    * Abstract class
    * Used as a combination of `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`

    >>> from torchmanager import Manager
    >>> class SomeDataset(Dataset):
    ...    @property
    ...    def unbatched_size(self) -> int: ...
    ...
    ...    def __getitem__(self, index: Any) -> Any: ...
    >>> dataset = SomeDataset(..., batch_size)
    >>> manager = Manager(...)
    >>> manager.fit(dataset, ...)

    - Properties:
        - batch_size: An `int` of batch size for the current dataset
        - device: A `torch.device` for the data to be pinned during iteration
        - drop_last: A `bool` flag of if drop the last data that not enought for the batch size
        - shuffle: A `bool` flag of if shuffling the data

    - Methods to implement:
        - unbatched_len: A property method that returns the total length of unbatched dataset
        - __get_item__: The built in method to get items by index (as in `torch.utils.data.Dataset`)
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

    @property
    def batched_len(self) -> int:
        if self.drop_last:
            return int(self.unbatched_len / self.batch_size)
        else:
            return math.ceil(self.unbatched_len / self.batch_size)

    def __init__(self, batch_size: int, /, *, device: torch.device = devices.CPU, drop_last: bool = False, shuffle: bool = False) -> None:
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

    def __iter__(self) -> Iterator[T]:
        # initialize devices
        cpu_count = os.cpu_count()
        if cpu_count is None:
            cpu_count = 0

        # initialize loader
        if self.device != devices.CPU:
            data_loader = DataLoader(self, batch_size=self.batch_size, drop_last=self.drop_last, shuffle=self.shuffle, num_workers=cpu_count, pin_memory=True, pin_memory_device=str(self.device))
        else:
            data_loader = DataLoader(self, batch_size=self.batch_size, drop_last=self.drop_last, shuffle=self.shuffle, num_workers=cpu_count)

        # yield data
        for data in data_loader:
            yield self.unpack_data(data)

    def __len__(self) -> int:
        """Returns the unbatched length"""
        return self.unbatched_len

    @staticmethod
    def unpack_data(data: Any) -> T:
        """
        Unpacks a single data into inputs and targets

        - Parameters:
            - data: `Any` kind of single data
        - Returns: `Any` kind of inputs with type `T`
        """
        if isinstance(data, torch.Tensor) or isinstance(data, dict):
            return data, data  # type: ignore # suppose for unsupervised reconstruction or a dictionary of packed data
        if isinstance(data, Sequence) and len(data) == 2:
            return data[0], data[1]  # type: ignore # suppose for supervised
        else:
            return NotImplemented  # unknown type of dataset


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
        assert not isinstance(loaded_dataset, Dataset), _raise(RuntimeError("The loaded dataset is a `torchmanager.data.Dataset` which has already been wrapped with batch loader during iteration."))
        data_loader = DataLoader(loaded_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory, num_workers=cpu_count, pin_memory_device=f"{targeted_devices[0].type}:{targeted_devices.index}")
        return data_loader
    return wrapped_fn
