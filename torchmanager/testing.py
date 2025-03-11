from torch.utils.data import DataLoader
from torchmanager_core import abc, devices, errors, torch, view
from torchmanager_core.protocols import Resulting
from torchmanager_core.typing import Any, Collection, Module

from .basic import BaseManager
from .data import Dataset


class BaseTestingManager(BaseManager[Module], abc.ABC):
    """
    A basic testing manager, only used for testing

    * extends: `.basic.BaseManager`
    * abstract methods: `test_step`

    - Properties:
        - compiled_metrics: The `dict` of metrics in `Resulting` that does not contain losses
        - summary: A `dict` of metrics summary with name in `str` and value in `float`
    """
    @property
    def compiled_metrics(self) -> dict[str, Resulting]:
        return {name: m for name, m in self.metric_fns.items() if "loss" not in name}

    @property
    def summary(self) -> dict[str, float]:
        # initialize
        summary: dict[str, float] = {}

        # summarize loss
        if self.loss_fn is not None:
            summary["loss"] = float(self.loss_fn.result.detach())

        # summarize metrics
        for name, fn in self.metric_fns.items():
            if name.startswith("val_") and self.model.training:
                continue
            else:
                name = name.replace("val_", "")
            try:
                summary[name] = float(fn.result.detach())
            except Exception as metric_error:
                runtime_error = errors.MetricError(name)
                raise runtime_error from metric_error
        return summary

    def eval(self, input: Any, target: Any, /) -> dict[str, float]:
        """
        Evaluate the model using metrics

        - Parameters:
            - input: `Any` kind of model output
            - target: `Any` kind of ground truth target
        - Returns: A `dict` of metrics summary with keys as name in `str` and values as metric value in `float`
        """
        # forward metrics
        for name, fn in self.compiled_metrics.items():
            if name.startswith("val_"):
                name = name.replace("val_", "")
            elif "loss" in name:
                continue
            try:
                _ = fn(input, target)
            except Exception as metric_error:
                runtime_error = errors.MetricError(name)
                raise runtime_error from metric_error
        return self.summary

    @torch.no_grad()
    def predict(self, dataset: DataLoader[Any] | Dataset[Any] | Collection[Any], /, *, device: torch.device | list[torch.device] | None = None, empty_cache: bool = True, use_multi_gpus: bool = False, show_verbose: bool = False) -> list[Any]:
        '''
        Predict the whole dataset

        - Parameters:
            - dataset: A `torch.utils.data.DataLoader` dataset to predict
            - device: An optional `torch.device` to test on if not using multi-GPUs or an optional default `torch.device` for testing otherwise
            - use_multi_gpus: A `bool` flag to use multi gpus during testing
            - show_verbose: A `bool` flag to show the progress bar during testing
        - Retruns: A `list` of `Any` prediction results
        '''
        # find available device
        cpu, device, target_devices = devices.search(device)
        if device == cpu or len(target_devices) < 2:
            use_multi_gpus = False
        devices.set_default(target_devices[0])

        # initialize dataset length
        if len(dataset) == 0:
            return []
        elif isinstance(dataset, Dataset):
            dataset_len = dataset.batched_len
        else:
            dataset_len = len(dataset)

        # initialize progress bar
        progress_bar = view.tqdm(total=dataset_len, disable=not show_verbose)

        # move model
        try:
            # multi gpus support
            if use_multi_gpus:
                use_multi_gpus = self.data_parallel(target_devices)

            # initialize predictions
            self.model.eval()
            predictions: list[Any] = []
            self.to(device)

            # loop the dataset
            for data in dataset:
                x, _ = self.unpack_data(data)
                if use_multi_gpus is not True:
                    x = devices.move_to_device(x, device)
                y, _ = self.forward(x)
                predictions.append(y)
                progress_bar.update()

            # reset model and loss
            return predictions
        except KeyboardInterrupt:
            view.logger.info("Prediction interrupted.")
            return []
        except Exception as error:
            view.logger.error(error)
            runtime_error = errors.PredictionError()
            raise runtime_error from error
        finally:
            # end epoch training
            progress_bar.close()

            # empty cache
            if empty_cache:
                self.reset(cpu)

    @torch.no_grad()
    def test(self, dataset: DataLoader[Any] | Dataset[Any] | Collection[Any], /, *, device: torch.device | list[torch.device] | None = None, empty_cache: bool = True, use_multi_gpus: bool = False, show_verbose: bool = False) -> dict[str, float]:
        """
        Test target model

        - Parameters:
            - dataset: A `torch.utils.data.DataLoader` or `.data.Dataset` dataset
            - device: An optional `torch.device` to test on if not using multi-GPUs or an optional default `torch.device` for testing otherwise
            - empyt_cache: A `bool` flag to empty cache after testing
            - use_multi_gpus: A `bool` flag to use multi gpus during testing
            - show_verbose: A `bool` flag to show the progress bar during testing
        - Returns: A `dict` of validation summary
        """
        # find available device
        cpu, device, target_devices = devices.search(device)
        if device == cpu or len(target_devices) < 2:
            use_multi_gpus = False
        devices.set_default(target_devices[0])

        # initialize dataset length
        if len(dataset) == 0:
            return {}
        elif isinstance(dataset, Dataset):
            dataset_len = dataset.batched_len
        else:
            dataset_len = len(dataset)

        # initialize progress bar
        progress_bar = view.tqdm(total=dataset_len, disable=not show_verbose)

        # reset loss function and metrics
        self.reset_metrics()

        try:
            # move to device
            if use_multi_gpus:
                use_multi_gpus = self.data_parallel(target_devices)
            self.to(device)

            # batch loop
            for data in dataset:
                # move x_test, y_test to device
                x_test, y_test = self.unpack_data(data)
                if use_multi_gpus is not True:
                    x_test = devices.move_to_device(x_test, device)
                y_test = devices.move_to_device(y_test, device)

                # test for one step
                step_summary = self.test_step(x_test, y_test)

                # update progress bar
                progress_bar.set_postfix(step_summary)
                progress_bar.update()

            # reset model and loss
            return self.summary
        except KeyboardInterrupt:
            view.logger.info("Testing interrupted.")
            return {}
        except Exception as error:
            view.logger.error(error)
            runtime_error = errors.TestingError()
            raise runtime_error from error
        finally:
            # end epoch training
            progress_bar.close()

            # empty cache
            if empty_cache:
                self.reset(cpu)

    @abc.abstractmethod
    def test_step(self, x_test: Any, y_test: Any) -> dict[str, float]:
        """
        A single testing step

        - Parameters:
            - x_train: The testing input in `torch.Tensor`
            - y_train: The testing label in `torch.Tensor`
        - Returns: A `dict` of validation summary
        """
        ...


class Manager(BaseTestingManager[Module]):
    """
    A testing manager, only used for testing

    * extends: `BaseTestingManager`

    Testing the model using `test` function:
    >>> from torchmanager.data import Dataset
    >>> dataset = Dataset(...)
    >>> manager.test(dataset, ...)
    """
    def test_step(self, x_test: Any, y_test: Any) -> dict[str, float]:
        """
        A single testing step

        - Parameters:
            - x_train: The testing data in `torch.Tensor`
            - y_train: The testing label in `torch.Tensor`
        - Returns: A `dict` of validation summary
        """
        # forward pass
        y, _ = self.forward(x_test, y_test)
        return self.eval(y, y_test)
