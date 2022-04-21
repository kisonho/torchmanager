from torchmanager.core import view
from . import train
from .managers import NightlyManager as Manager, clone

view.warnings.warn("[Beta warning]: This package includes features for future version, all modules in this package are not stable.", FutureWarning)

version = "1.1.0a2"