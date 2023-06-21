from .callback import Callback, FrequencyCallback
from .ckpt import BestCheckpoint, LastCheckpoint, MonitorType
from .dynamic import DynamicWeight, LambdaDynamicWeight
from .early_stop import EarlyStop
from .lr import LrSchedueler
from torchmanager_core.protocols import Frequency
Checkpoint = LastCheckpoint

try:
    from .experiment import Experiment
    from .tensorboard import TensorBoard
except:
    from torchmanager_core import view
    view.warnings.warn("Tensorboard dependency is not installed, install it to use `Experiment` and `TensorBoard` callbacks.", ImportWarning)
    Experiment = TensorBoard = NotImplemented
    pass
