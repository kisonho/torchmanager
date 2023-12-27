from .configs import ConfigsFormatError
from .runtime import PredictionError, TestingError, TransformError
from .train import LossError, MetricError, StopTraining
from .version import VersionError

def _raise(e: Exception) -> None:
    raise e