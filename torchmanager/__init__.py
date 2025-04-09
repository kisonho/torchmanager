from torchmanager_core import VERSION
from torchmanager_core.view import VerboseType

from . import callbacks, configs, data, losses, metrics
from .testing import BaseTestingManager, Manager as TestingManager
from .training import BaseTrainingManager, Manager as TrainingManager

Manager = TrainingManager
version = VERSION
