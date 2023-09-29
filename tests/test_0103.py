import torch, torchmanager, torchmanager_core
from torchmanager_core import Version, API_VERSION
from unittest import TestCase


class Test0103(TestCase):
    callback_fn_called: bool

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

    def test_wrapped_fn(self) -> None:
        from torchmanager.losses import loss, loss_fn
        from torchmanager.metrics import metric, metric_fn
        from torchmanager_core.protocols import WrappedFn

        # check loss wrapper
        @loss
        def some_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.mean(input - target)
        self.assertIsInstance(some_loss, WrappedFn, "Wrapped function is not a `WrappedFn`.")

        # check loss_fn wrapper
        @loss_fn(target="target", weight=1)
        def some_loss_fn(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.mean(input - target)
        self.assertIsInstance(some_loss_fn, WrappedFn, "Wrapped function is not a `WrappedFn`.")

        # check metric wrapper
        @metric
        def some_metric(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.mean(input - target)
        self.assertIsInstance(some_metric, WrappedFn, "Wrapped function is not a `WrappedFn`.")

        # check metric_fn wrapper
        @metric_fn(target="target")
        def some_metric_fn(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.mean(input - target)
        self.assertIsInstance(some_metric_fn, WrappedFn, "Wrapped function is not a `WrappedFn`.")
