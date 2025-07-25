from torch.utils.data import Dataset as _Dataset, DataLoader, Sampler
from torchmanager_core import abc, devices, errors, math, os, torch, _raise
from torchmanager_core.typing import Any, Callable, Iterable, Iterator, Optional, Sequence, TypeVar, cast

D = TypeVar("D", bound=DataLoader)
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
    num_workers: int
    sampler: Sampler[list[T]] | Iterable[list[T]] | None
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

    def __init__(self, batch_size: int, /, *, device: torch.device = devices.CPU, drop_last: bool = False, num_workers: Optional[int] = os.cpu_count(), sampler: Sampler[list[T]] | Iterable[list[T]] | None = None, shuffle: bool = False) -> None:
        """
        Constructor

        - Parameters:
            - batch_size: An `int` of batch size for the current dataset
            - device: A `torch.device` for the data to be pinned during iteration
            - drop_last: A `bool` flag of if drop the last data that not enought for the batch size
            - num_workers: An optional `int` of the number of cpus to load the data
            - sampler: An optional `torch.utils.data.Sampler` or `Iterable` of `list` of indices
            - shuffle: A `bool` flag of if shuffling the data
        """
        super().__init__()
        self.__device = device
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.shuffle = shuffle

        # initialize num workers
        if num_workers is None:
            cpu_count = os.cpu_count()
            self.num_workers = 0 if cpu_count is None else cpu_count
        else:
            self.num_workers = num_workers

    def __contains__(self, value: Any) -> bool:
        for i in range(len(self)):
            if self[i] == value:
                return True
        return False

    @abc.abstractmethod
    def __getitem__(self, index: Any) -> Any:
        """Returns an unbatched item"""
        return NotImplemented

    def __iter__(self) -> Iterator[tuple[T, T]]:
        # initialize loader
        if self.device != devices.CPU:
            data_loader = DataLoader(self, batch_size=self.batch_size, drop_last=self.drop_last, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=True, pin_memory_device=str(self.device), sampler=self.sampler)
        else:
            data_loader = DataLoader(self, batch_size=self.batch_size, drop_last=self.drop_last, shuffle=self.shuffle, num_workers=self.num_workers, sampler=self.sampler)

        # yield data
        for data in data_loader:
            yield self.unpack_data(data)

    def __len__(self) -> int:
        """Returns the unbatched length"""
        return self.unbatched_len

    @staticmethod
    def unpack_data(data: Any) -> tuple[T, T]:
        """
        Unpacks a single data into inputs and targets

        - Parameters:
            - data: `Any` kind of single data
        - Returns: A `tuple` of `Any` kind of inputs with type `T`
        """
        if isinstance(data, torch.Tensor) or isinstance(data, dict):
            return cast(T, data), cast(T, data)  # suppose for unsupervised reconstruction or a dictionary of packed data
        if isinstance(data, Sequence) and len(data) == 2:
            return data[0], data[1]  # suppose for supervised
        else:
            return NotImplemented  # unknown type of dataset


class PreprocessedDataset(Dataset[T], abc.ABC):
    """
    A data with preprocessing methods

    * extends: `Dataset`
    * Abstract class

    - Properties:
        - transforms: An `Iterable` of `Callable` preprocessing function that returns `Any` kind of preprocessed object.

    - Methods to implement:
        - transforms: A property methods that returns a list of `Callable` preprocessing function
        - load: The method to load an item by index without preprocessing
    """

    @property
    @abc.abstractmethod
    def transforms(self) -> Iterable[Callable[..., Any]]:
        return NotImplemented

    def __getitem__(self, index: Any) -> Any:
        # load data
        data = self.load(index)

        # preprocess transforms
        for fn in self.transforms:
            try:
                data = fn(data)
            except Exception as e:
                raise errors.TransformError(fn, data) from e
        return data

    @abc.abstractmethod
    def load(self, index: Any) -> Any:
        """
        The method to load a raw item by index without preprocessing

        - Parameters:
            - index: `Any` kind of index object
        - Returns: `Any` kind of non-preprocessed object
        """
        return NotImplemented


def batched(fn: Callable[..., _Dataset], loader_type: type[D] = DataLoader) -> Callable[..., D]:
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

    Use with specific data loader type:
    >>> from torch.utils.data import DataLoader
    >>> class CustomizedDataLoader(DataLoader):
    ...     ...
    >>> @batched(loader_type=CustomizedDataLoader)
    >>> def load_some_dataset(...) -> TorchDataset:
    ...     ...
    """
    # wrap function
    def wrapped_fn(*args: Any, batch_size: int = 1, device: torch.device = devices.CPU, drop_last: bool = False, num_workers: Optional[int] = None, shuffle: bool = False, **kwargs: Any) -> D:
        # initialize devices
        if num_workers is None:
            cpu_count = os.cpu_count()
            num_workers = 0 if cpu_count is None else cpu_count
        _, device, targeted_devices = devices.search(device)
        if device == devices.CPU:
            pin_memory = False
        else:
            devices.set_default(targeted_devices[0])
            pin_memory = True

        # load dataset
        loaded_dataset = fn(*args, **kwargs)
        assert not isinstance(loaded_dataset, Dataset), _raise(RuntimeError("The loaded dataset is a `torchmanager.data.Dataset` which has already been wrapped with batch loader during iteration."))
        data_loader = loader_type(loaded_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory, pin_memory_device=f"{targeted_devices[0].type}:{targeted_devices.index}")
        return data_loader
    return wrapped_fn
