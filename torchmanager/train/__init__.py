from torchmanager_core import view, deprecated

from .checkpoint import Checkpoint, list_checkpoints, load
from .learning_rate import update_lr

view.warnings.warn("The `torchmanager.train` package will be deprecated from v1.3 and will be removed from v1.4.", PendingDeprecationWarning)