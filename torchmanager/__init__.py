from torchmanager_core import VERSION
from torchmanager_core.view import VerboseType

from . import callbacks, configs, data, losses, metrics
from .testing import Manager as TestingManager
from .training import Manager as TrainingManager

Manager = TrainingManager
version = VERSION
