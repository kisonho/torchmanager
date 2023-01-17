from .train import LossError, MetricError, StopTraining

def _raise(e: Exception) -> None:
    raise e