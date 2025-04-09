from torchmanager_core.typing import Generic, NamedTuple, TypeVar


I = TypeVar('I')
T = TypeVar('T')


class DataPair(NamedTuple, Generic[I, T]):
    """
    A data pair for training and testing

    * Used in `torchmanager.data.pair`

    >>> from torchmanager import DataPair
    >>> pair = DataPair(train_data, test_data)
    >>> print(pair.train_data)
    >>> print(pair.test_data)
    """
    input: I
    target: T
