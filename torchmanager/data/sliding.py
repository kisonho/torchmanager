from itertools import product

from torchmanager_core import torch, _raise

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
        # calculate the number of windows
        num_windows = (dim_size - window_dim) // stride_dim + 1
        window_dims.append(num_windows)
        stride_dims.append(stride_dim)

        # add the very last indices when the window size is not divisible by the stride
        if (dim_size - window_dim) % stride_dim != 0:
            window_dims[-1] += 1

    # Iterate over each window
    window_starts = list(product(*[range(num_windows) for num_windows in window_dims]))
    for indices in window_starts:
        # Calculate the starting coordinates of the window and adjust the very last indices when the window size is not divisible by the stride
        indices = (slice(None),) + tuple(slice(min(i * s, image.shape[dim + 1] - ws), min(i * s+ws, image.shape[dim + 1])) for dim, (i, s, ws) in enumerate(zip(indices, stride, window_size)))

        # Extract the window from the image
        window = image[indices]

        # Add the window to the list
        windows.append(window.unsqueeze(0))
    return torch.cat(windows)

def reversed_sliding_window(windows: torch.Tensor, /, image_size: tuple[int, ...], stride: tuple[int, ...]) -> torch.Tensor:
    """
    Reverses the sliding window operation on input windows.

    Parameters:
        - windows: A a `torch.Tensor` with shape [w, c, *window_size].
        - image_size: A `tuple` of the size of the original image without channel size in `int`.
        - stride: A `tuple` of the stride of the sliding window in `int`.
    Returns: A reconstructed image in `torch.Tensor` with shape [c, *image_size].
    """
    # Get dimensions
    _, num_channels, *window_size = windows.shape
    stride_dims: list[int] = []
    window_dims: list[int] = []
    
    # Calculate output shape
    output_shape = (num_channels, *image_size)

    # Calculate number of windows in each dimension
    for dim_size, window_dim, stride_dim in zip(output_shape[1:], window_size, stride):
        num_windows = (dim_size - window_dim) // stride_dim + 1
        window_dims.append(num_windows)
        stride_dims.append(stride_dim)

        # add the very last indices when the window size is not divisible by the stride
        if (dim_size - window_dim) % stride_dim != 0:
            window_dims[-1] += 1

    # Initialize output tensor
    output = torch.zeros(output_shape, dtype=windows.dtype, device=windows.device)
    overlap = torch.zeros(output_shape, dtype=torch.int, device=windows.device)
    window_starts = list(product(*[range(num_windows) for num_windows in window_dims]))
    assert len(window_starts) == windows.shape[0], _raise(ValueError(f"Number of windows ({windows.shape[0]}) must match the number of windows calculated ({len(window_starts)})."))

    # Iterate over windows
    for i, indices in enumerate(window_starts):
        # Calculate the starting coordinates of the window
        indices = (slice(None),) + tuple(slice(min(i * s, image_size[dim] - ws), min(i * s+ws, image_size[dim])) for dim, (i, s, ws) in enumerate(zip(indices, stride, window_size)))

        # Place the window into output tensor
        output[indices] += windows[i]
        overlap[indices] += 1

    # average the overlap of output tensor
    overlap = torch.clamp(overlap, min=1)
    output /= overlap
    return output
