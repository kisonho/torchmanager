import torch, torchmanager, torchmanager_core
from torchmanager_core import Version, API_VERSION
from typing import Optional
from unittest import TestCase


class Test0101(TestCase):
    def test_backward_wrapper(self) -> None:
        # compile manager
        model = torch.nn.Linear(100, 100)
        manager = torchmanager.Manager(model)
        x = torch.randn((64, 100))
        y, _ = manager.forward(x, None)
        assert isinstance(y, torch.Tensor)

        # wrapper function
        @manager.backward_fn
        def mse_backward(loss: torch.Tensor) -> None:
            manager.current_epoch += 1
            loss -= x
            loss = loss.abs().mean()
            loss.backward()

        # backward
        manager.backward(y)
        self.assertEqual(manager.current_epoch, 1)

    def test_forward_wrapper(self) -> None:
        # compile manager
        model = torch.nn.Linear(100, 3)
        manager = torchmanager.Manager(model)
        x = torch.randn((64, 100))
        y, _ = manager.forward(x, None)
        self.assertEqual(y.shape, torch.Size([64, 3]))

        # wrapper function
        @manager.forward_fn
        def softmax_forward(input: torch.Tensor, target: Optional[torch.Tensor], /) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            y: torch.Tensor = manager.model(input)
            y = y.softmax(dim=1)
            if manager.loss_fn is not None and target is not None:
                loss = manager.compiled_losses(y, target)
            else:
                loss = None
            return y.argmax(dim=1), loss

        # forward
        y, _ = manager.forward(x, None)
        assert isinstance(y, torch.Tensor)
        self.assertEqual(y.shape, torch.Size([64]))

    def test_version(self) -> None:
        self.assertGreaterEqual(API_VERSION, "1.3")

    def test_py_version(self) -> None:
        import platform

        py_version = Version(platform.python_version())
        self.assertGreaterEqual(py_version, "3.9")
