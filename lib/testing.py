from torchmanager_core import devices, torch, view, _raise
from torchmanager_core.typing import Any, Dict, Generic, Module, Optional, SizedIterable
from torchmanager_core.view import warnings

from .losses import Loss
from .metrics import Metric
from .basic import DataManager, BaseManager

class Manager(BaseManager, DataManager, Generic[Module]):
    """
    A testing manager, only used for testing

    * extends: `.basic.BaseManager`, `.basic.DataManager`

    - Properties:
        - compiled_losses: The loss function in `Loss` that must be exist
        - compiled_metrics: The `dict` of metrics in `Metric` that does not contain losses
    """
    model: Module
    
    @property
    def compiled_losses(self) -> Loss:
        assert self.loss_fn is not None,  _raise(NotImplementedError("[Training Error]: loss_fn is not given, compiles the manager with loss_fn first."))
        warnings.warn("The compiled_losses property in a `TestingManager` will be deprecated from v1.1.0 and will be removed from v1.2.0.", PendingDeprecationWarning)
        return self.loss_fn

    @property
    def compiled_metrics(self) -> Dict[str, Metric]:
        return {name: m for name, m in self.metric_fns.items() if "loss" not in name}

    def test(self, dataset: Any, device: Optional[torch.device] = None, use_multi_gpus: bool = False, show_verbose: bool = False) -> Dict[str, float]:
        """
        Test target model

        - Parameters:
            - dataset: Either `SizedIterable` or `data.DataLoader` to load the dataset
            - device: An optional `torch.device` to test on
            - use_multi_gpus: A `bool` flag to use multi gpus during testing
            - show_verbose: A `bool` flag to show the progress bar during testing
        - Returns: A `dict` of validation summary
        """
        # arguments checking
        assert isinstance(dataset, SizedIterable), _raise(ValueError("The dataset must be both Sized and Iterable."))

        # initialize function
        if self.loss_fn is not None: self.loss_fn.reset()
        for _, m in self.metric_fns.items(): m.reset()

        # find available device
        cpu, device = devices.find(device)

        # multi gpu support
        if use_multi_gpus is True:
            if not isinstance(self.model, torch.nn.parallel.DataParallel):
                raw_model = self.model
                self.model = torch.nn.parallel.DataParallel(self.model)
            else: raw_model = None

            if not isinstance(self.loss_fn, torch.nn.parallel.DataParallel) and self.loss_fn is not None:
                raw_loss_fn = self.loss_fn
                paralleled_loss_fn = torch.nn.parallel.DataParallel(self.loss_fn)
                self.loss_fn = Loss(paralleled_loss_fn)
            else: raw_loss_fn = None
        else:
            raw_model = None
            raw_loss_fn = None

        # set module status
        try:
            self.model.eval()
            self.to(device)
        except: pass

        # initialize progress bar
        if len(dataset) == 0: return {}
        progress_bar = view.tqdm(total=len(dataset)) if show_verbose else None

        # disable auto gradients
        with torch.no_grad():
            # batch loop
            for data in dataset:
                # move x_test, y_test to device
                x_test, y_test = self.unpack_data(data)
                if use_multi_gpus is not True and isinstance(x_test, torch.Tensor):
                    x_test = devices.move_to_device(x_test, device)
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