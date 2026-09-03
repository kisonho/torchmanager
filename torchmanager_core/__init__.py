import abc, argparse, copy, gc, json, math, os, platform, shutil, sys, torch
from torch.nn import functional
from torch.utils import data

from . import backward, checkpoint, devices, errors, random, version, view
from .errors import raise_error
from .version import Version, deprecated, API as API_VERSION, CURRENT as VERSION, DESCRIPTION

try:
    from torch.utils import tensorboard
except ImportError:
    view.warnings.warn("Module tensorboard is not installed.", ImportWarning)
    tensorboard = NotImplemented

try:
    import yaml
except ImportError:
    view.warnings.warn("Module yaml is not installed.", ImportWarning)
    yaml = NotImplemented

__all__ = [
    "abc",
    "argparse",
    "copy",
    "gc",
    "json",
    "math",
    "os",
    "platform",
    "shutil",
    "sys",
    "torch",
    "functional",
    "data",
    "backward",
    "checkpoint",
    "devices",
    "errors",
    "random",
    "version",
    "view",
    "tensorboard",
    "yaml",
    "Version",
    "VERSION",
    "API_VERSION",
    "DESCRIPTION",
    "deprecated",
    "raise_error"
]

@deprecated("v1.5", "v2.0")
def _raise(e: Exception) -> None:
    raise_error(e)
