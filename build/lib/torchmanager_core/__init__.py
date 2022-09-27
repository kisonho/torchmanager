import abc, gc, math, os, sys, torch
from torch.nn import functional
from torch.utils import data

from . import devices, view
from .version import deprecated, CURRENT_VERSION as VERSION
try: from torch.utils import tensorboard
except ImportError:
    view.warnings.warn("[Core Warning]: Module tensorboard is not installed.", ImportWarning)
    tensorboard = NotImplemented

def _raise(e: Exception) -> None:
    raise e