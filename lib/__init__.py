import torchmanager_core as core
from torchmanager_core.view import VerboseType

from . import callbacks, core, losses, metrics, train
from .testing import Manager as TestingManager
from .training import Manager as TrainingManager
Manager = TrainingManager

version = "1.0.5b2"