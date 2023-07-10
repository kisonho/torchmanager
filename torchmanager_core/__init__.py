import abc, argparse, gc, math, os, platform, shutil, sys, torch
from torch.nn import functional
from torch.utils import data

from . import checkpoint, devices, errors, random, view
from .errors import _raise
from .version import Version, deprecated, API as API_VERSION, CURRENT as VERSION, DESCRIPTION

try:
    from torch.utils import tensorboard
except ImportError:
    view.warnings.warn("Module tensorboard is not installed.", ImportWarning)
    tensorboard = NotImplemented

if Version(platform.python_version()) < "v3.9":
    view.warnings.warn("Torchmanager will no longer support Python 3.8 from v1.3.", PendingDeprecationWarning)
