from torchmanager_core.typing import Any, NamedTuple


class DataPair(NamedTuple):
    """
    A data pair for training and testing

    * Used in `torchmanager.data.pair`

    >>> from torchmanager import DataPair
    >>> pair = DataPair(train_data, test_data)
    >>> print(pair.train_data)
    >>> print(pair.test_data)
    """
    input: Any
    target: Any
