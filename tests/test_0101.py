import torch
from torchmanager import callbacks, losses
from torchmanager_core import errors, Version
from unittest import TestCase


class Test0101(TestCase):
    def test_dice_loss(self) -> None:
        # initialize
        print("Testing Dice Loss")
        loss_fn = losses.Dice()
        y = torch.randn((4, 4, 224, 224))
        y_test = torch.randn_like(y).argmax(1)

        # calculate dice loss
        loss = loss_fn(y, y_test)
        self.assertGreater(float(loss), 0, "Loss must be a positive value, got {loss}.")

    def test_early_stopping(self) -> None:
        # initialize
        n = 0
        accs = [0.3, 0.2, 0.5, 0.4, 0.6, 0.65, 0.5, 0.5, 0.4, 0.4, 0.34, 0.34]
        early_stopping = callbacks.EarlyStop("acc", steps=3)

        # training simulation
        for i in range(10):
            # on epoch end
            try:
                early_stopping.on_epoch_end(i, summary={"acc": accs[i]})
                n += 1
            except errors.StopTraining:
                break

        self.assertEqual(n, 9)

    def test_py_version(self) -> None:
        import platform

        py_version = Version(platform.python_version())
        self.assertGreaterEqual(py_version, "3.8")
