from torchmanager_core.protocols import LrSteping
from torchmanager_core.typing import Dict


def update_lr(lr_scheduler: LrSteping) -> Dict[str, float]:
    """
    Update lr scheduler and returns the current learning rate as a summary

    - Parameters:
        - lr_scheduler: A `torch.optim.lr_scheduler._LRScheduler` to update
    - Returns: A `dict` of lr summary
    """
    # update lr
    lr_scheduler.step()
    lr_list = lr_scheduler.get_last_lr()
    lr_summary: Dict[str, float] = {}

    # update summary
    if len(lr_list) > 1:
        for i, lr in enumerate(lr_list):
            lr_summary[f"lr_{i}"] = lr
    else:
        lr_summary["lr"] = lr_list[0]
    return lr_summary
