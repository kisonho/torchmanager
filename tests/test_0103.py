import torch, torchmanager, torchmanager_core
from torchmanager_core import Version, API_VERSION
from typing import Optional
from unittest import TestCase


class Test0103(TestCase):
    backward_fn_called: bool
    callback_fn_called: bool
    forward_fn_called: bool

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
            self.backward_fn_called = True
            manager.current_epoch += 1
            loss -= x
            loss = loss.abs().mean()
            loss.backward()

        # backward
        self.backward_fn_called = False
        manager.backward(y)
        self.assertEqual(manager.current_epoch, 1)
        self.assertTrue(self.backward_fn_called, "Backward function not called")

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
            self.forward_fn_called = True
            y: torch.Tensor = manager.model(input)
            y = y.softmax(dim=1)
            if manager.loss_fn is not None and target is not None:
                loss = manager.compiled_losses(y, target)
            else:
                loss = None
            return y.argmax(dim=1), loss

        # forward
        self.forward_fn_called = False
        y, _ = manager.forward(x, None)
        assert isinstance(y, torch.Tensor)
        self.assertEqual(y.shape, torch.Size([64]))
        self.assertTrue(self.forward_fn_called, "Forward function not called")

    def test_functional_callbacks(self) -> None:
        from torchmanager.callbacks import LambdaCallback, on_epoch_start, on_batch_end, on_batch_start, on_epoch_end

        @on_batch_end
        def on_batch_end_callback(batch_index: int, summary: dict[str, float]) -> None:
            self.callback_fn_called = True
            self.assertGreaterEqual(batch_index, 0, f"Epoch index must be a non-negative number, got {batch_index}.")

        @on_batch_start
        def on_batch_start_callback(batch_index: int) -> None:
            self.callback_fn_called = True
            self.assertGreaterEqual(batch_index, 0, f"Epoch index must be a non-negative number, got {batch_index}.")

        @on_epoch_end
        def on_epoch_end_callback(epoch_index: int, summary: dict[str, float]) -> None:
            self.callback_fn_called = True
            self.assertGreaterEqual(epoch_index, 0, f"Epoch index must be a non-negative number, got {epoch_index}.")

        @on_epoch_start
        def on_epoch_start_callback(epoch_index: int) -> None:
            self.callback_fn_called = True
            self.assertGreaterEqual(epoch_index, 0, f"Epoch index must be a non-negative number, got {epoch_index}.")

        self.callback_fn_called = False
        on_batch_end_callback.on_batch_end(0)
        self.assertTrue(self.callback_fn_called, "On batch end function not called")
        self.assertIsInstance(on_batch_end_callback, LambdaCallback, "On batch end function is not a `LambdaCallback`.")

        self.callback_fn_called = False
        on_batch_start_callback.on_batch_start(0)
        self.assertTrue(self.callback_fn_called, "On batch start function not called")
        self.assertIsInstance(on_batch_start_callback, LambdaCallback, "On batch start function is not a `LambdaCallback`.")

        self.callback_fn_called = False
        on_epoch_end_callback.on_epoch_end(0)
        self.assertTrue(self.callback_fn_called, "On epoch end function not called")
        self.assertIsInstance(on_epoch_end_callback, LambdaCallback, "On epoch end function is not a `LambdaCallback`.")

        self.callback_fn_called = False
        on_epoch_start_callback.on_epoch_start(0)
        self.assertTrue(self.callback_fn_called, "On epoch start function not called")
        self.assertIsInstance(on_epoch_start_callback, LambdaCallback, "On epoch start function is not a `LambdaCallback`.")

    def test_py_version(self) -> None:
        import platform

        py_version = Version(platform.python_version())
        self.assertGreaterEqual(py_version, "3.9")

    def test_version(self) -> None:
        self.assertGreaterEqual(API_VERSION, "1.3")
