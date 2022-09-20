from random import shuffle
from torch.utils.data import IterableDataset, DataLoader
from torchmanager_core import abc, devices, os, torch
from torchmanager_core.typing import Any, Iterator, Sequence, Tuple

class Dataset(IterableDataset, abc.ABC):
    '''
    A dataset that iterates with batch size

    * extends: `IterableDataset`
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
    '''
    __device: torch.device
    batch_size: int
    drop_last: bool
    shuffle: bool

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
        '''
        super().__init__()
        self.__device = device
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    @abc.abstractmethod
    def __getitem__(self, index: Any) -> Any:
        return NotImplemented

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        # initialize devices
        cpu_count = os.cpu_count()
        if cpu_count is None: cpu_count = 0
        device = self.device

        # yield data
        data_loader = DataLoader(self, batch_size=self.batch_size, drop_last=self.drop_last, shuffle=self.shuffle, num_workers=cpu_count, pin_memory=(device == devices.CPU))
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