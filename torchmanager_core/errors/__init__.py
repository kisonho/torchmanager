from .configs import ConfigsFormatError
from .runtime import PredictionError, TestingError, TransformError
from .train import LossError, MetricError, StopTraining
from .version import VersionError

__all__ = (
    "ConfigsFormatError",
    "PredictionError",
    "TestingError",
    "TransformError",
    "LossError",
    "MetricError",
    "StopTraining",
    "VersionError",
    "raise_error"
)

def raise_error(e: Exception) -> None:
    raise e

_raise = raise_error
