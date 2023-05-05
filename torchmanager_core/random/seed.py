import random, torch

try:
    import numpy as np  # type: ignore
except ImportError:
    np = NotImplemented

from .. import devices

def freeze_seed(seed: int, /) -> None:
    """
    Freeze random with given seed

    - Parameters:
        - seed: An `int` for the random seed
    """
    random.seed(seed)
    if np is not NotImplemented:
        np.random.seed(seed)
    torch.manual_seed(seed)

    if devices.GPU is not NotImplemented:
        torch.cuda.manual_seed_all(seed)

def unfreeze_seed() -> int:
    """
    Regenerate a random seed

    - Returns: An `int` of random seed
    """
    # random PyTorch seed
    seed_64 = torch.random.seed()
    
    # set 32 bit seed for random and np package
    seed_32 = int(seed_64 / (2 ** 32)) if seed_64 > 2 ** 32 - 1 else seed_64 # convert to 32 bit seed
    random.seed(seed_32)
    if np is not NotImplemented:
        np.random.seed(seed_32)

    # set cuda seed
    if devices.GPU is not NotImplemented:
        torch.cuda.manual_seed_all(seed_64)
    return seed_64
