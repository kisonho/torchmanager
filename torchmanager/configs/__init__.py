from .basic import Configs
from .json import JSONConfigs

try:
    from .yaml import YAMLConfigs
except ImportError:
    from torchmanager_core import view
    view.warnings.warn("PyYAML dependency is not installed, install it to use `YAMLConfigs`.", ImportWarning)
    YAMLConfigs = NotImplemented
