from torchmanager_core import torch, _raise
from torchmanager_core.typing import Dict, Enum, Optional
from torchmanager_core.view import logging
from torchmanager_core.view.verbose import _VerboseControllable

class LrScheduleFreq(Enum):
    """The frequency enum of learning rate schedule"""
    EPOCH = 0
    BATCH = 1

def initial_step_lr_scheduler(lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], initial_epoch: int = 0) -> None:
    """
    Initialize learning rate scheduler for the initial epochs before training starts

    - Parameters:
        - lr_scheduler: The given lr scheduler in `torch.optim.lr_scheduler._LRScheduler`
        - initial_epoch: An `int` of the intial epoch index
    """
    # go to initial epoch
    if lr_scheduler is not None and initial_epoch > 0:
        # disable verbose
        assert isinstance(lr_scheduler, _VerboseControllable), _raise(TypeError("lr_scheduler does not performs to the VerboseControllable protocol."))
        verbose = lr_scheduler.verbose
        lr_scheduler.verbose = False
        logging.info(f"Intializing learning rate with {initial_epoch} epochs...")

        # steps to initial epoch
        for _ in range(initial_epoch):
            lr_scheduler.step()

        # reset verbose
        lr_scheduler.verbose = verbose

def update_lr(lr_scheduler: torch.optim.lr_scheduler._LRScheduler) -> Dict[str, float]:
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
            lr_summary[f'lr_{i}'] = lr
    else: lr_summary['lr'] = lr_list[0]
    return lr_summary
