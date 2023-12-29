import torch
from torchmanager import metrics, Manager
from unittest import TestCase


class Test0100(TestCase):
    def test_manager(self) -> None:
        # initialize model
        model = torch.nn.Sequential(torch.nn.Linear(16, 16))

        # initialize manager
        manager = Manager(model)

        # test manager
        self.assertIsInstance(manager, Manager)

    def test_metrics(self) -> None:
        # initialize
        metric_fn = metrics.Accuracy()

        # calculate accuracy
        for i in range(10):
            y = torch.randint(0, 10, (i+1,1,1,1))
            y_test = torch.randint(0, 10, (i+1,1,1,1))
            metric_fn(y, y_test)
        self.assertGreaterEqual(float(metric_fn.result), 0, "Accuracy must be a positive value, got {loss}.")

    def test_py_version(self) -> None:
        import platform
        major, minor, _ = platform.python_version_tuple()
        self.assertGreaterEqual(int(major), 3)
        self.assertGreaterEqual(int(minor), 8)
