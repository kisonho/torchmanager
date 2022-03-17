from ..core import torch, view
from ..core._typing import Dict, Optional

def initial_step_lr_scheduler(lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], initial_epoch: int = 0) -> None:
    # go to initial epoch
    if lr_scheduler is not None and initial_epoch > 0:
        # disable verbose
        assert isinstance(lr_scheduler, view._VerboseControllable), "[Runtime Error]: lr_scheduler does not performs to the VerboseControllable protocol."
        verbose = lr_scheduler.verbose
        lr_scheduler.verbose = False

        # steps to initial epoch
        for _ in range(initial_epoch):
            lr_scheduler.step()

        # reset verbose
        lr_scheduler.verbose = verbose

def update_lr(lr_scheduler: torch.optim.lr_scheduler._LRScheduler) -> Dict[str, float]:
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
