from torch.utils.data import DataLoader
from torchmanager_core import devices, errors, math, torch, view
from torchmanager_core.protocols import Resulting
from torchmanager_core.typing import Any, Collection, Dict, List, Module, Optional, Self, Union

from .callbacks import Callback
from .data import Dataset
from .losses import Loss, ParallelLoss
from .metrics import Metric
from .train import Checkpoint, update_lr
from .testing import Manager as _Manager


class Manager(_Manager[Module]):
    """
    A training manager

    * extends: `.testing.Manager`
    * [Deprecation Warning]: Method `train` has been set as protected from v1.0.2, the public method will be removed from v1.2.0. Override `_train` method instead.

    Train using fit method:
    >>> from torchmanager.data import Dataset
    >>> dataset = Dataset(...)
    >>> epochs: int = ...
    >>> manager.fit(dataset, epochs, ...)

    - Properties:
        - current_epoch: The `int` index of current training epoch
        - compiled_optimizer: The `torch.optim.Optimizer` that must be exist
    """
    __current_epoch: int

    @property
    def current_epoch(self) -> int:
        return self.__current_epoch

    @current_epoch.setter
    def current_epoch(self, e: int) -> None:
        if e < 0:
            raise ValueError(f"The epoch index must be a non_negative integer, got {e}.")
        else:
            self.__current_epoch = e

    @property
    def compiled_losses(self) -> Resulting:
        assert self.loss_fn is not None, errors._raise(NotImplementedError("The manager is not compiled properly, `loss_fn` is missing."))
        return self.loss_fn

    @property
    def compiled_optimizer(self) -> torch.optim.Optimizer:
        assert self.optimizer is not None, errors._raise(NotImplementedError("The manager is not compiled properly, `optimizer` is missing."))
        return self.optimizer

    def __init__(self, model: Module, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[Union[Loss, Dict[str, Loss]]] = None, metrics: Dict[str, Metric] = {}) -> None:
        super().__init__(model, optimizer, loss_fn, metrics)
        self.__current_epoch = 0

    def _train(self, dataset: Union[DataLoader[Any], Dataset[Any], Collection], iterations: Optional[int] = None, device: torch.device = devices.CPU, use_multi_gpus: bool = False, show_verbose: bool = False, verbose_type: view.VerboseType = view.VerboseType.ALL, callbacks_list: List[Callback] = []) -> Dict[str, float]:
        """
        The single training step for an epoch

        - Parameters:
            - dataset: A `torch.utils.data.DataLoader` or `.data.Dataset` training dataset
            - iterations: An optional `int` of total training iterations, must be smaller than the size of dataset
            - device: A `torch.device` where the data is moved to, should be same as the model
            - use_multi_gpus: A `bool` flag of if using multi gpus
            - show_verbose: A `bool` flag of if showing progress bar
            - verbose_type: A `view.VerboseType` that controls the display of verbose
            - callbacks_list: A `list` of callbacks in `Callback`
        - Returns: A summary of `dict` with keys as `str` and values as `float`
        """
        # initialize status
        self.model.train()
        self.compiled_losses.train()
        for m in self.compiled_metrics.values():
            m.train()

        # reset loss and metrics
        self.compiled_losses.reset()
        for m in self.metric_fns.values():
            m.reset()

        # run deprecated method
        summary = self.train(dataset, device=device, use_multi_gpus=use_multi_gpus, show_verbose=show_verbose, callbacks_list=callbacks_list)
        if summary is not NotImplemented:
            view.warnings.warn("Method `train` has been set to protected from v1.0.2 and will be removed in v1.2.0, override `_train` instead.", DeprecationWarning)
            return summary

        # initialize progress bar
        iterations = len(dataset) if iterations is None else iterations
        progress_bar = view.tqdm(total=iterations) if show_verbose else None

        # batch loop
        for batch, data in enumerate(dataset):
            # on batch start
            for c in callbacks_list:
                c.on_batch_start(batch)

            # move x_train and y_train to device
            x_train, y_train = self.unpack_data(data)
            if use_multi_gpus is not True:
                x_train = devices.move_to_device(x_train, device)
            y_train = devices.move_to_device(y_train, device)

            # train for one step
            summary = self.train_step(x_train, y_train)

            # on batch start
            for c in callbacks_list:
                c.on_batch_end(batch, summary=summary)

            # implement progress bar
            if progress_bar is not None:
                # initialize progress summary
                if verbose_type == view.VerboseType.LOSS:
                    progress_summary = {name: s for name, s in summary.items() if "loss" in name}
                elif verbose_type == view.VerboseType.METRICS:
                    progress_summary = {name: s for name, s in summary.items() if "loss" not in name}
                elif verbose_type == view.VerboseType.ALL:
                    progress_summary = summary
                else:
                    raise TypeError(f"Verbose type {verbose_type} is not supported.")

                # update progress bar
                progress_bar.set_postfix(progress_summary)
                progress_bar.update()

            # check for iterations
            if batch + 1 >= iterations:
                break

        # end epoch training
        if progress_bar is not None:
            progress_bar.close()

        # summarize
        summary = {name: float(fn.result.detach()) for name, fn in self.metric_fns.items() if not name.startswith("val_")}
        summary["loss"] = float(self.compiled_losses.result.detach())
        return summary

    def fit(self, training_dataset: Union[DataLoader[Any], Dataset[Any], Collection], epochs: Optional[int] = None, iterations: Optional[int] = None, initial_epoch: Optional[int] = None, lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, val_dataset: Optional[Union[DataLoader[Any], Dataset[Any], Collection]] = None, device: Optional[Union[torch.device, list[torch.device]]] = None, use_multi_gpus: bool = False, callbacks_list: List[Callback] = [], **kwargs) -> torch.nn.Module:
        """
        Training algorithm

        - Parameters:
            - training_dataset: Any kind of training dataset in `torch.utils.data.DataLoader` or `.data.Dataset`
            - epochs: An optional `int` number of training epochs
            - iterations: An optional `int` number of training iterations
            - lr_scheduelr: An optioanl `torch.optim.lr_scheduler._LRScheduler` to update the lr per epoch
            - val_dataset: An optional validation `Any`
            - device: An optional `torch.device` to test on if not using multi-GPUs or an optional default `torch.device` for testing otherwise
            - use_multi_gpus: A `bool` flag of if using multi gpus
            - callbacks_list: A `list` of callbacks in `Callback`
            - **kwargs: Additional keyword arguments that will be passed to `train` method.
        - Returns: A trained `torch.nn.Module`
        """
        # arguments checking
        assert self.compiled is True, errors._raise(ValueError("Manager has not yet been compiled. Either loss_fn or optimizer, or both, are not given."))
        if epochs is not None:
            assert epochs > 0, errors._raise(ValueError(f"The epochs must be a positive integer, got {epochs}."))
            assert iterations is None, errors._raise(ValueError(f"The iterations must be given as `None` when epochs is given, got {iterations}."))
        else:
            assert iterations is not None, errors._raise(ValueError(f"The iterations must be given if epochs is not given."))
            assert iterations > 0, errors._raise(ValueError(f"The iterations must be a positive integer, got {iterations}."))
            assert epochs is None, errors._raise(ValueError(f"The epochs must be given as `None` when iterations is given, got {epochs}."))
            epochs = math.ceil(iterations / len(training_dataset))

        # initialize initial epoch
        if initial_epoch is not None:
            assert initial_epoch >= 0, errors._raise(ValueError(f"The initial_epoch must be a non_negative integer, got {initial_epoch}."))
            assert initial_epoch < epochs, errors._raise(ValueError(f"The initial_epoch must be smaller than total epochs, got epochs={epochs} but initial_epoch={initial_epoch}."))
            self.current_epoch = initial_epoch
        elif self.current_epoch > 0:
            initial_epoch = self.current_epoch + 1  # skip the latest current epoch when resuming training
        else:
            initial_epoch = self.current_epoch

        # initialize training
        cpu, device, target_devices = devices.search(device)
        if device == cpu and len(target_devices) < 2:
            use_multi_gpus = False
        devices.set_default(target_devices[0])
        if lr_scheduler is not None:
            view.warnings.warn("Parameter `lr_scheduler` has been deprecated after v1.1.0 and will be removed from v1.2.0, use `.callbacks.LrScheduler` callback instead.", DeprecationWarning)
        for c in callbacks_list:
            c.on_train_start(initial_epoch)

        # multi gpus support for model
        if use_multi_gpus and not isinstance(self.model, torch.nn.parallel.DataParallel):
            model, use_multi_gpus = devices.data_parallel(self.model, devices=target_devices)
        else:
            model = self.model
        self.model = model

        # multi gpus support for loss
        if use_multi_gpus and not isinstance(self.compiled_losses, torch.nn.parallel.DataParallel):
            assert isinstance(self.compiled_losses, Loss), errors._raise(TypeError("The compiled loss function is not a valid `Loss` object."))
            paralleled_loss_fn, use_multi_gpus = devices.data_parallel(self.compiled_losses, devices=target_devices, parallel_type=ParallelLoss)
            assert isinstance(paralleled_loss_fn, ParallelLoss) or isinstance(paralleled_loss_fn, Loss), errors._raise(TypeError("Paralleled function is not a valid `ParallelLoss` or `Loss` after parallel."))
            self.loss_fn = paralleled_loss_fn
        self.to(device)

        # epoch loop
        for self.current_epoch in range(initial_epoch, epochs):
            # initialize epoch
            view.logger.info(f"Training epoch {self.current_epoch + 1}/{epochs}")
            for c in callbacks_list:
                c.on_epoch_start(self.current_epoch)
            if iterations is not None:
                batch_iterations = iterations if len(training_dataset) < iterations else iterations
            else:
                batch_iterations = None

            # train for one epoch
            summary = self._train(training_dataset, iterations=batch_iterations, device=device, use_multi_gpus=use_multi_gpus, callbacks_list=callbacks_list, **kwargs)
            if iterations is not None and batch_iterations is not None:
                iterations -= batch_iterations

            # validate
            val_summary = self.test(val_dataset, device=device, use_multi_gpus=use_multi_gpus, empty_cache=False) if val_dataset is not None else {}

            # on epoch end
            for c in callbacks_list:
                try:
                    c.on_epoch_end(self.current_epoch, summary=summary, val_summary=val_summary)
                except errors.StopTraining:
                    # on train end
                    for c in callbacks_list:
                        c.on_train_end(self.raw_model)
                    self.model = self.raw_model.to(cpu)
                    self.loss_fn = self.raw_loss_fn.to(cpu) if self.raw_loss_fn is not None else self.raw_loss_fn
                    devices.empty_cache()
                    return self.model
                except Exception:
                    raise

            # step lr scheduler
            if lr_scheduler is not None:
                lr_summary = update_lr(lr_scheduler)
                summary.update(lr_summary)

            # print summary info
            val_message = f"Epoch {self.current_epoch + 1}/{epochs}: "
            summary.update({f"val_{name}": value for name, value in val_summary.items()})
            for i, (name, value) in enumerate(summary.items()):
                if i > 0:
                    val_message += ", "
                val_message += f"{name}={value:.4f}"
            view.logger.info(val_message)

        # on train end
        for c in callbacks_list:
            c.on_train_end(self.raw_model)
        self.model = self.raw_model.to(cpu)
        self.loss_fn = self.raw_loss_fn.to(cpu) if self.raw_loss_fn is not None else self.raw_loss_fn
        devices.empty_cache()
        return self.model

    def train(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        """The single training step for an epoch"""
        return NotImplemented

    def train_step(self, x_train: Any, y_train: Any) -> Dict[str, float]:
        """
        A single training step

        - Parameters:
            - x_train: The training data
            - y_train: The training label
        - Returns: A summary of `dict` with keys as `str` and values as `float`
        """
        # forward pass
        summary: Dict[str, float] = {}
        y = self.model(x_train)
        loss = self.compiled_losses(y, y_train)

        # forward metrics
        for name, fn in self.compiled_metrics.items():
            if not name.startswith("val_") and "loss" not in name:
                fn(y, y_train)

        # backward pass
        self.compiled_optimizer.zero_grad()
        loss.backward()
        self.compiled_optimizer.step()

        # summary result
        try:
            summary["loss"] = float(self.compiled_losses.result.detach())
        except Exception as e:
            raise RuntimeError("Cannot fetch loss.") from e
        for name, fn in self.metric_fns.items():
            if name.startswith("val_"):
                continue
            try:
                summary[name] = float(fn.result.detach())
            except Exception as metric_error:
                runtime_error = RuntimeError(f"Cannot fetch metric '{name}'.")
                raise runtime_error from metric_error
        return summary

    def to_checkpoint(self) -> Checkpoint[Self]:
        ckpt = super().to_checkpoint()
        ckpt.last_epoch = self.current_epoch
        return ckpt
