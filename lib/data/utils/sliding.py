from itertools import product
from torchmanager_core import torch, _raise
from torchmanager_core.typing import List, Tuple


def sliding_window(image: torch.Tensor, window_size: Tuple[int, ...], stride: Tuple[int, ...]) -> torch.Tensor:
    """
    Extract sliding windows from a multi-dimensional `torch.Tensor`.

    - Parameters:
        - image: The input image `torch.Tensor`. Can have any number of dimensions.
        - window_size: A `tuple` of the size of the sliding window in `int` to extract from the image, must have the same number of dimensions as the input image tensor.
        - stride: A `tuple` of the stride of the sliding window in `int`, must have the same number of dimensions as the input image tensor.
    Returns: A `list` of `torch.Tensor`, where each tensor corresponds to a sliding window extracted from the input image tensor.
    Raises: `ValueError` if the window size or stride are not valid for the input image tensor.

    Example:
    ```
    >>> image = torch.randn(3, 224, 224)
    >>> window_size = (64, 64)
    >>> stride = (32, 32)
    >>> windows = sliding_window(image, window_size, stride)
    >>> windows.shape[0]
    49
    ```
    """
    # initialize
    assert len(window_size) == len(image.shape) == len(image.shape), _raise(ValueError(f"Window size dimension ({len(window_size)}) and stride dimension ({stride}) must be the same of image ({len(image.shape)})."))
    stride_dims: List[int] = []
    windows: List[torch.Tensor] = []
    window_dims: List[int] = []

    # Calculate number of windows in each dimension
    for dim_size, window_dim, stride_dim in zip(image.shape, window_size, stride):
        num_windows = (dim_size - window_dim) // stride_dim + 1
        window_dims.append(num_windows)
        stride_dims.append(stride_dim)

    # Iterate over each window
    window_starts = product(*[range(num_windows) for num_windows in window_dims])
    for starts in window_starts:
        # Calculate the starting coordinates of the window
        starts = torch.tensor(starts)
        window_starts = starts * torch.tensor(stride_dims)
        window_ends = window_starts + torch.tensor(window_size)

        # Extract the window from the image
        window = image[tuple(slice(s, e) for s, e in zip(window_starts, window_ends))]

        # Add the window to the list
        windows.append(window.unsqueeze(0))
    return torch.cat(windows)
