import numpy as np, random, torch

from torchmanager_core import devices

def freeze_seed(seed: int) -> None:
    """
    Freeze random with given seed

    - Parameters:
        - seed: An `int` for the random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if devices.GPU is not NotImplemented:
        torch.cuda.manual_seed_all(seed)

def unfreeze_seed() -> int:
    """
    Regenerate a random seed

    - Returns: An `int` of random seed
    """
    seed = torch.random.seed()
    random.seed()
    np.random.seed()

    if devices.GPU is not NotImplemented:
        torch.cuda.manual_seed_all(seed)
    return seed
