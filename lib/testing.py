from torchmanager_core import devices, torch, view, _raise, deprecated
from torchmanager_core.typing import Any, Collection, Dict, Generic, List, Module, Optional, Union
from torchmanager_core.view import warnings

from .losses import Loss, ParallelLoss
from .metrics import Metric
from .basic import BaseManager

class Manager(BaseManager[Module], Generic[Module]):
    """
    A testing manager, only used for testing

    * extends: `.basic.BaseManager`

    Testing the model using `test` function:
    >>> from torchmanager.data import Dataset
    >>> dataset = Dataset(...)
    >>> manager.test(dataset, ...)

    - Properties:
        - compiled_losses: The loss function in `Loss` that must be exist
        - compiled_metrics: The `dict` of metrics in `Metric` that does not contain losses
    """
    model: Union[Module, torch.nn.parallel.DataParallel]
    
    @property
    @deprecated("1.1.0", "1.2.0")
    def compiled_losses(self) -> Union[Loss, ParallelLoss]:
        assert self.loss_fn is not None,  _raise(NotImplementedError("loss_fn is not given, compiles the manager with loss_fn first."))
        return self.loss_fn

    @property
    def compiled_metrics(self) -> Dict[str, Metric]:
        return {name: m for name, m in self.metric_fns.items() if "loss" not in name}

    @torch.no_grad()
    def predict(self, dataset: Collection, device: Optional[Union[torch.device, list[torch.device]]] = None, use_multi_gpus: bool = False, show_verbose: bool = False) -> List[Any]:
        '''
        Predict the whole dataset

        - Parameters:
            - dataset: A `Collection` dataset to predict
            - device: An optional `torch.device` to test on if not using multi-GPUs or an optional default `torch.device` for testing otherwise
            - use_multi_gpus: A `bool` flag to use multi gpus during testing
            - show_verbose: A `bool` flag to show the progress bar during testing
        '''
        # find available device
        cpu, device, target_devices = devices.search(None if use_multi_gpus else device)
        if device == cpu and len(target_devices) < 2: use_multi_gpus = False
        devices.set_default(target_devices[0])

        # move model
        if use_multi_gpus and not isinstance(self.model, torch.nn.parallel.DataParallel):
            raw_model = self.model
            self.model, use_multi_gpus = devices.data_parallel(self.model, devices=target_devices)
        else: raw_model = None

        # initialize predictions
        self.model.eval()
        predictions: List[Any] = []
        if len(dataset) == 0: return predictions
        progress_bar = view.tqdm(total=len(dataset)) if show_verbose else None

        # loop the dataset
        for data in dataset:
            x, _ = self.unpack_data(data)
            if use_multi_gpus is not True: x = devices.move_to_device(x, device)
            y = self.model(x)
            predictions.append(y)
            if progress_bar is not None: progress_bar.update()

        # reset model and loss
        if raw_model is not None: self.model = raw_model.to(cpu)
        devices.empty_cache()
        return predictions

    @torch.no_grad()
    def test(self, dataset: Collection, device: Optional[Union[torch.device, list[torch.device]]] = None, use_multi_gpus: bool = False, show_verbose: bool = False) -> Dict[str, float]:
        """
        Test target model

        - Parameters:
            - dataset: A `Collection` dataset
            - device: An optional `torch.device` to test on if not using multi-GPUs or an optional default `torch.device` for testing otherwise
            - use_multi_gpus: A `bool` flag to use multi gpus during testing
            - show_verbose: A `bool` flag to show the progress bar during testing
        - Returns: A `dict` of validation summary
        """
        # find available device
        cpu, device, target_devices = devices.search(None if use_multi_gpus else device)
        if device == cpu and len(target_devices) < 2: use_multi_gpus = False
        devices.set_default(target_devices[0])

        # move model
        if use_multi_gpus and not isinstance(self.model, torch.nn.parallel.DataParallel):
            raw_model = self.model
            self.model, use_multi_gpus = devices.data_parallel(self.model, devices=target_devices)
        else: raw_model = None

        # move loss function
        if use_multi_gpus and self.loss_fn is not None and not isinstance(self.loss_fn, torch.nn.parallel.DataParallel):
            raw_loss_fn = self.loss_fn
            paralleled_loss_fn, use_multi_gpus = devices.data_parallel(self.loss_fn, devices=target_devices, parallel_type=ParallelLoss)
            if use_multi_gpus: self.loss_fn = Loss(paralleled_loss_fn)
        else: raw_loss_fn = None

        # set module status
        self.model.eval()
        if self.loss_fn is not None: self.loss_fn.eval().reset()
        for _, m in self.metric_fns.items(): m.eval().reset()
        self.to(device)

        # initialize progress bar
        if len(dataset) == 0: return {}
        progress_bar = view.tqdm(total=len(dataset)) if show_verbose else None

        # batch loop
        for data in dataset:
            # move x_test, y_test to device
            x_test, y_test = self.unpack_data(data)
            if use_multi_gpus is not True: x_test = devices.move_to_device(x_test, device)
            y_test = devices.move_to_device(y_test, device)

            # test for one step
            step_summary = self.test_step(x_test, y_test)

            # implement progress bar
            if progress_bar is not None:
                progress_bar.set_postfix(step_summary)
                progress_bar.update()

        # end epoch training
        if progress_bar is not None:
            progress_bar.close()
        
        # summarize
        summary: Dict[str, float] = {}
        for name, fn in self.metric_fns.items():
            if name.startswith("val_"): name = name.replace("val_", "")
            try: summary[name] = float(fn.result.detach())
            except Exception as metric_error:
                runtime_error = RuntimeError(f"Cannot fetrch metric '{name}'.")
                raise runtime_error from metric_error
        if self.loss_fn is not None: summary["loss"] = float(self.loss_fn.result.detach())

        # reset model and loss
        if raw_model is not None: self.model = raw_model.to(cpu)
        if raw_loss_fn is not None: self.loss_fn = raw_loss_fn.to(cpu)
        devices.empty_cache()
        return summary

    def test_step(self, x_test: Any, y_test: Any) -> Dict[str, float]:
        """
        A single testing step

        - Parameters:
            - x_train: The testing data in `torch.Tensor`
            - y_train: The testing label in `torch.Tensor`
        - Returns: A `dict` of validation summary
        """
        # initialize
        summary: Dict[str, float] = {}

        # forward pass
        y = self.model(x_test)

        # forward metrics
        for name, fn in self.compiled_metrics.items():
            if name.startswith("val_"): name = name.replace("val_", "")
            elif "loss" in name: continue
            try:
                fn(y, y_test)
                summary[name] = float(fn.result.detach())
            except Exception as metric_error:
                runtime_error = RuntimeError(f"Cannot fetch metric '{name}'.")
                raise runtime_error from metric_error

        # forward loss
        if self.loss_fn is not None:
            self.loss_fn(y, y_test)
            summary["loss"] = float(self.loss_fn.result.detach())
        return summary