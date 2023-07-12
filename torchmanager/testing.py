from torch.utils.data import DataLoader
from torchmanager_core import devices, errors, torch, view
from torchmanager_core.protocols import Resulting
from torchmanager_core.typing import Any, Callable, Collection, Module, Optional, Union

from .basic import BaseManager
from .data import Dataset


class Manager(BaseManager[Module]):
    """
    A testing manager, only used for testing

    * extends: `.basic.BaseManager`

    Testing the model using `test` function:
    >>> from torchmanager.data import Dataset
    >>> dataset = Dataset(...)
    >>> manager.test(dataset, ...)

    - Properties:
        - compiled_metrics: The `dict` of metrics in `Resulting` that does not contain losses
        - summary: A `dict` of metrics summary with name in `str` and value in `float`
    """
    model: Union[Module, torch.nn.parallel.DataParallel]

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
            elif name.startswith("val_"):
                name = name.replace("val_", "")
            try:
                summary[name] = float(fn.result.detach())
            except Exception as metric_error:
                runtime_error = errors.MetricError(name)
                raise runtime_error from metric_error
        return summary

    def forward(self, input: Any, target: Optional[Any] = None, /) -> tuple[Any, Optional[torch.Tensor]]:
        """
        Forward pass function

        - Parameters:
            - x_train: The training data
        - Returns: `Any` kind of model output
        """
        # forward model
        y = self.model(input)

        # forward loss
        if self.loss_fn is not None and target is not None:
            try:
                loss = self.loss_fn(y, target)
            except Exception as loss_error:
                runtime_error = errors.LossError()
                raise loss_error from runtime_error
        else:
            loss = None
        return y, loss

    def forward_fn(self, fn: Callable[[Any, Optional[Any]], tuple[Any, Optional[torch.Tensor]]], /) -> None:
        """
        The wrapper function to override `forward` method

        * Example:

        ```
        >>> manager = Manager(...) # define a manager

        >>> @manager.forward_fn
        >>> def forward(input: Any, target: Optional[Any]) -> tuple[Any, Optional[torch.Tensor]]: # the forward function to override
        >>>     y = manager.model(input)
        >>>     loss = manager.compiled_loss(y, target)
        >>>     return y, loss

        >>> ...
        >>> manager.fit(...)
        ```

        - Parameters:
            - fn: A forward function that accepts `Any` type of `x` and optional `Any` type of `y` and returns a `tuple` of `Any` kind of model output and an optional `torch.Tensor` loss
        """
        setattr(self, "forward", fn)

    @torch.no_grad()
    def predict(self, dataset: Union[DataLoader[Any], Dataset[Any], Collection[Any]], /, *, device: Optional[Union[torch.device, list[torch.device]]] = None, use_multi_gpus: bool = False, show_verbose: bool = False) -> list[Any]:
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
        if device == cpu and len(target_devices) < 2:
            use_multi_gpus = False
        devices.set_default(target_devices[0])

        # initialize
        if len(dataset) == 0:
            return []
        elif isinstance(dataset, Dataset):
            dataset_len = dataset.batched_len
        else:
            dataset_len = len(dataset)
        progress_bar = view.tqdm(total=dataset_len) if show_verbose else None

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
                y, _ = self.forward(x, None)
                predictions.append(y)
                if progress_bar is not None:
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
            if progress_bar is not None:
                progress_bar.close()

            # empty cache
            self.reset(cpu)

    @torch.no_grad()
    def test(self, dataset: Union[DataLoader[Any], Dataset[Any], Collection[Any]],  /, *,device: Optional[Union[torch.device, list[torch.device]]] = None, empty_cache: bool = True, use_multi_gpus: bool = False, show_verbose: bool = False) -> dict[str, float]:
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
        if device == cpu and len(target_devices) < 2:
            use_multi_gpus = False
        devices.set_default(target_devices[0])

        # initialize progress bar
        if len(dataset) == 0:
            return {}
        elif isinstance(dataset, Dataset):
            dataset_len = dataset.batched_len
        else:
            dataset_len = len(dataset)
        progress_bar = view.tqdm(total=dataset_len) if show_verbose else None

        # reset loss function and metrics
        if self.loss_fn is not None:
            self.loss_fn.eval().reset()
        for _, m in self.metric_fns.items():
            m.eval().reset()

        try:
            # move to device
            if use_multi_gpus:
                use_multi_gpus = self.data_parallel(target_devices)
            self.to(device)
            self.model.eval()

            # batch loop
            for data in dataset:
                # move x_test, y_test to device
                x_test, y_test = self.unpack_data(data)
                if use_multi_gpus is not True:
                    x_test = devices.move_to_device(x_test, device)
                y_test = devices.move_to_device(y_test, device)

                # test for one step
                step_summary = self.test_step(x_test, y_test)

                # implement progress bar
                if progress_bar is not None:
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
            if progress_bar is not None:
                progress_bar.close()

            # empty cache
            if empty_cache:
                self.reset(cpu)

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

        # forward metrics
        for name, fn in self.compiled_metrics.items():
            if name.startswith("val_"):
                name = name.replace("val_", "")
            elif "loss" in name:
                continue
            try:
                fn(y, y_test)
            except Exception as metric_error:
                runtime_error = errors.MetricError(name)
                raise runtime_error from metric_error
        return self.summary
