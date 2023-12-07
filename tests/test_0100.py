import torch
from torchmanager import metrics
from torchmanager_core import errors, Version
from unittest import TestCase


class Test0100(TestCase):
    def test_metrics(self) -> None:
        # initialize
        metric_fn = metrics.Accuracy()

        # calculate accuracy
        for i in range(10):
            y = torch.randint(0, 10, (i+1,))
            y_test = torch.randint(0, 10, (i+1,))
            metric_fn(y, y_test)
        self.assertGreaterEqual(float(metric_fn.result), 0, "Accuracy must be a positive value, got {loss}.")

    def test_py_version(self) -> None:
        import platform

        py_version = Version(platform.python_version())
        self.assertGreaterEqual(py_version, "3.8")
