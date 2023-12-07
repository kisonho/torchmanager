from .basic import Configs
from .json import JSONConfigs

try:
    from .yaml import YAMLConfigs
except ImportError:
    YAMLConfigs = NotImplemented
