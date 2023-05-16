from torch.utils.data import DataLoader
from torchmanager_core import devices, errors, torch, view, _raise
from torchmanager_core.protocols import Resulting
from torchmanager_core.typing import Any, Collection, Dict, List, Module, Optional, Union

from .basic import BaseManager
from .data import Dataset
from .losses import Loss, ParallelLoss


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
    def compiled_metrics(self) -> Dict[str, Resulting]:
        return {name: m for name, m in self.metric_fns.items() if "loss" not in name}

    @property
    def summary(self) -> Dict[str, float]:
        # initialize
        summary: Dict[str, float] = {}

        # summarize loss
        if self.loss_fn is not None:
            summary["loss"] = float(self.loss_fn.result.detach())

        # summarize metrics
        for name, fn in self.metric_fns.items():
            if name.startswith("val_"):
                name = name.replace("val_", "")
            try:
                summary[name] = float(fn.result.detach())
            except Exception as metric_error:
                runtime_error = errors.MetricError(name)
                raise runtime_error from metric_error
        return summary


    def forward(self, x_train: Any) -> Any:
        """
        Forward pass function

        - Parameters:
            - x_train: The training data
        - Returns: `Any` kind of model output
        """
        return self.model(x_train)

    @torch.no_grad()
    def predict(self, dataset: Union[DataLoader[Any], Dataset[Any], Collection[Any]], /, *, device: Optional[Union[torch.device, List[torch.device]]] = None, use_multi_gpus: bool = False, show_verbose: bool = False) -> List[Any]:
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
            if use_multi_gpus and not isinstance(self.model, torch.nn.parallel.DataParallel):
                self.model, use_multi_gpus = devices.data_parallel(self.model, devices=target_devices)

            # initialize predictions
            self.model.eval()
            predictions: List[Any] = []
            self.to(device)

            # loop the dataset
            for data in dataset:
                x, _ = self.unpack_data(data)
                if use_multi_gpus is not True:
                    x = devices.move_to_device(x, device)
                y = self.forward(x)
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
            self.model = self.raw_model.to(cpu)
            devices.empty_cache()

    @torch.no_grad()
    def test(self, dataset: Union[DataLoader[Any], Dataset[Any], Collection[Any]],  /, *,device: Optional[Union[torch.device, List[torch.device]]] = None, empty_cache: bool = True, use_multi_gpus: bool = False, show_verbose: bool = False) -> Dict[str, float]:
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
        # initialize device
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
            # multi-gpus support for model
            if use_multi_gpus and not isinstance(self.model, torch.nn.parallel.DataParallel):
                self.model, use_multi_gpus = devices.data_parallel(self.model, devices=target_devices)

            # multi-gpus support for loss function
            if use_multi_gpus and self.loss_fn is not None and not isinstance(self.loss_fn, torch.nn.parallel.DataParallel):
                paralleled_loss_fn, use_multi_gpus = devices.data_parallel(self.loss_fn, devices=target_devices, parallel_type=ParallelLoss)
                assert isinstance(paralleled_loss_fn, ParallelLoss) or isinstance(paralleled_loss_fn, Loss), _raise(TypeError("Paralleled function is not a valid `ParallelLoss` or `Loss` after parallel."))
                self.loss_fn = paralleled_loss_fn

            # set module status and move to device
            self.model.eval()
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
                self.model = self.raw_model.to(cpu)
                self.loss_fn = self.raw_loss_fn.to(cpu) if self.raw_loss_fn is not None else self.raw_loss_fn
                devices.empty_cache()

    def test_step(self, x_test: Any, y_test: Any) -> Dict[str, float]:
        """
        A single testing step

        - Parameters:
            - x_train: The testing data in `torch.Tensor`
            - y_train: The testing label in `torch.Tensor`
        - Returns: A `dict` of validation summary
        """
        # forward pass
        y = self.forward(x_test)

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

        # forward loss
        if self.loss_fn is not None:
            try:
                self.loss_fn(y, y_test)
            except Exception as loss_error:
                runtime_error = errors.LossError()
                raise loss_error from runtime_error
        return self.summary
