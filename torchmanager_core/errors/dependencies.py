from typing import Any


class WithoutScipy:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(f"Please install `scipy` and use `torchmanager_scipy` plugin for this callback.")


class WithoutTensorboard:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(f"Please install `tensorboard` and use `torchmanager_tensorboard` plugin for this callback.")
