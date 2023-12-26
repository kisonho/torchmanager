from itertools import product

from torchmanager_core import torch, _raise
from torchmanager_core.typing import Union

def sliding_window(image: torch.Tensor, /, window_size: tuple[int, ...], stride: tuple[int, ...]) -> torch.Tensor:
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
    36
    ```
    """
    # initialize
    assert len(window_size) == len(stride), _raise(ValueError(f"Window size dimension ({len(window_size)}) and stride dimension ({stride}) must be the same."))
    stride_dims: list[int] = []
    windows: list[torch.Tensor] = []
    window_dims: list[int] = []

    # Calculate number of windows in each dimension
    for dim_size, window_dim, stride_dim in zip(image.shape[1:], window_size, stride):
        num_windows = (dim_size - window_dim) // stride_dim + 1
        window_dims.append(num_windows)
        stride_dims.append(stride_dim)

    # Iterate over each window
    window_starts = product(*[range(num_windows) for num_windows in window_dims])
    for indices in window_starts:
        # Calculate the starting coordinates of the window
        indices = (slice(None),) + tuple(slice(i, i+ws) for i, ws in zip(indices, window_size))

        # Extract the window from the image
        window = image[indices]

        # Add the window to the list
        windows.append(window.unsqueeze(0))
    return torch.cat(windows)

def reversed_sliding_window(windows: Union[torch.Tensor, list[torch.Tensor]], /, image_size: tuple[int, ...]) -> torch.Tensor:
    """
    Reverses the sliding window operation on input windows.

    Parameters:
        - windows: A list of `torch.Tensor` with shape [b, c, *window_size] or a `torch.Tensor` with shape [b, w, c, *window_size].
        - image_size: A `tuple` of the size of the original image without channel size in `int`.
    Returns: A reconstructed image in `torch.Tensor` with shape [b, c, *image_size].
    """
    # unpack windows, convert a full tensor with shape [b, w, c, *window_size] into list of windows tensor with shape [b, c, *window_size]
    list_of_windows = [windows[:, i, ...] for i in range(windows.shape[1])] if isinstance(windows, torch.Tensor) else windows

    # Get batch size, number of windows, and window shape
    num_windows = len(list_of_windows)
    batch_size, _, *window_shape = list_of_windows[0].size()

    # Initialize the output tensor with zeros
    output = torch.zeros((batch_size, *image_size), dtype=list_of_windows[0].dtype, device=list_of_windows[0].device)

    # Calculate the step size along each dimension
    step_sizes = [size // num_windows for size in image_size]

    # Iterate through each window and place it in the corresponding position in the output tensor
    for i in range(num_windows):
        # Calculate the start and end indices for each dimension
        start_indices = [i * step for i, step in zip(range(len(window_shape)), step_sizes)]
        end_indices = [(i + 1) * step for i, step in zip(range(len(window_shape)), step_sizes)]

        # Create slices for each dimension
        slices = tuple(slice(start, end) for start, end in zip(start_indices, end_indices))

        # Place the window in the corresponding position in the output tensor
        output[:, :, slices] = list_of_windows[i]
    return output
