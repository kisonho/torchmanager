from torch.utils.data import IterableDataset, DataLoader as _Loader
from torchmanager_core import abc, devices, os, torch
from torchmanager_core.typing import Any, Iterator, Sequence, Tuple

class Dataset(IterableDataset, abc.ABC):
    '''
    A dataset that iterates with batch size

    * extends: `IterableDataset`
    * implements: `typing.Collection`
    * Abstract class
    * Used as a combination of `torch.utils.data.IterableDataset` and `torch.utils.data.DataLoader`
    
    >>> from torchmanager import Manager
    >>> class SomeDataset(Dataset):
    ...    def __init__(self, ...,  batch_size: int, device: torch.device = devices.CPU) -> None: ...
    ...    def __getitem__(self, index: Any) -> Any: ...
    ...    def __len__(self, index: Any) -> Any: ...
    >>> dataset = SomeDataset(..., batch_size)
    >>> manager = Manager(...)
    >>> manager.fit(dataset, ...)

    - Properties:
        - batch_size: An `int` of batch size for the current dataset
        - device: A `torch.device` for the data to be pinned during iteration
        - drop_last: A `bool` flag of if drop the last data that not enought for the batch size
        - shuffle: A `bool` flag of if shuffling the data
    '''
    __batch_size: int
    __device: torch.device
    drop_last: bool
    shuffle: bool

    @property
    def batch_size(self) -> int: return self.__batch_size

    @batch_size.setter
    def batch_size(self, b: int) -> None:
        if b <= 0: raise ValueError('Batch size must be a positive number')
        self.__batch_size = b

    @property
    def device(self) -> torch.device:
        _, device, target_devices = devices.search(self.__device)
        devices.set_default(target_devices[0])
        return device

    def __init__(self, batch_size: int, device: torch.device = devices.CPU, drop_last: bool = False, shuffle: bool = False) -> None:
        '''
        Constructor

        - Parameters:
            - batch_size: An `int` of batch size for the current dataset
            - device: A `torch.device` for the data to be pinned during iteration
            - drop_last: A `bool` flag of if drop the last data that not enought for the batch size
            - shuffle: A `bool` flag of if shuffling the data
        '''
        super().__init__()
        self.__device = device
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __contains__(self, value: Any) -> bool:
        for i in range(len(self)):
            if self[i] == value: return True
        return False

    @abc.abstractmethod
    def __getitem__(self, index: Any) -> Any:
        return NotImplemented

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        # initialize devices
        cpu_count = os.cpu_count()
        if cpu_count is None: cpu_count = 0
        device = self.device

        # yield data
        data_loader = _Loader(self, batch_size=self.batch_size, drop_last=self.drop_last, shuffle=self.shuffle, num_workers=cpu_count, pin_memory=(device == devices.CPU))
        for data in data_loader: yield self.unpack_data(data)

    @abc.abstractmethod
    def __len__(self) -> int: return NotImplemented

    @staticmethod
    def unpack_data(data: Any) -> Tuple[Any, Any]:
        '''
        Unpacks a single data into inputs and targets
        
        - Parameters:
            - data: `Any` kind of single data
        - Returns: A `tuple` of `Any` kind of inputs and `Any` kind of targets
        '''
        if isinstance(data, Sequence):
            return data[0], data[1] if len(data) >= 2 else NotImplemented
        else: return NotImplemented

class DataLoader(_Loader):
    '''
    A PyTorch `DataLoader` that performs to `typing.Collection` protocol
    '''
    def __contains__(self, value: Any) -> bool:
        for element in self:
            if value == element: return True
        return False