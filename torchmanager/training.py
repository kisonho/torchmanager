from torch.utils.data import DataLoader
from torchmanager_core import devices, errors, math, torch, view, _raise
from torchmanager_core.checkpoint import Checkpoint
from torchmanager_core.protocols import Resulting
from torchmanager_core.typing import Any, Collection, Module, Optional, Self, Union, overload

from .callbacks import Callback, ProgressBar
from .data import Dataset
from .losses import Loss
from .metrics import Metric
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

    def __init__(self, model: Module, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[Union[Loss, dict[str, Loss]]] = None, metrics: dict[str, Metric] = {}) -> None:
        super().__init__(model, optimizer, loss_fn, metrics)
        self.__current_epoch = 0

    def _train(self, dataset: Union[DataLoader[Any], Dataset[Any], Collection], /, iterations: Optional[int] = None, *, device: torch.device = devices.CPU, use_multi_gpus: bool = False, callbacks_list: list[Callback] = [], **kwargs: Any) -> dict[str, float]:
        """
        The single training step for an epoch

        - Parameters:
            - dataset: A `torch.utils.data.DataLoader` or `.data.Dataset` training dataset
            - iterations: An optional `int` of total training iterations, must be smaller than the size of dataset
            - device: A `torch.device` where the data is moved to, should be same as the model
            - use_multi_gpus: A `bool` flag of if using multi gpus
            - callbacks_list: A `list` of callbacks in `Callback`
        - Returns: A summary of `dict` with keys as `str` and values as `float`
        """
        # initialize status
        self.model = self.model.train()
        self.loss_fn = self.compiled_losses.train()
        for k, m in self.compiled_metrics.items():
            self.compiled_metrics[k] = m.train()

        # reset loss and metrics
        self.compiled_losses.reset()
        for m in self.metric_fns.values():
            m.reset()

        # initialize iterations
        dataset_len = dataset.batched_len if isinstance(dataset, Dataset) else len(dataset)
        iterations = dataset_len if iterations is None else iterations

        # batch loop
        for batch, data in enumerate(dataset):
            # on batch start
            for callback in callbacks_list:
                callback.on_batch_start(batch)

            # move x_train and y_train to device
            x_train, y_train = self.unpack_data(data)
            if use_multi_gpus is not True:
                x_train = devices.move_to_device(x_train, device)
            y_train = devices.move_to_device(y_train, device)

            # train for one step
            summary = self.train_step(x_train, y_train)

            # on batch end
            for callback in callbacks_list:
                callback.on_batch_end(batch, summary=summary)

            # check for iterations
            if batch + 1 >= iterations:
                break
        return self.summary

    def backward(self, loss: torch.Tensor, /) -> None:
        """
        Backward function to calculate the gradients
        
        - Parameters:
            - loss: A `torch.Tensor` of loss value
        """
        loss.backward()

    @overload
    def fit(self, training_dataset: Union[DataLoader[Any], Dataset[Any], Collection], /, epochs: int, val_dataset: Optional[Union[DataLoader[Any], Dataset[Any], Collection]] = None, callbacks_list: list[Callback] = [], *, iterations: None = None, initial_epoch: Optional[int] = None, return_summary: bool = False, device: Optional[Union[torch.device, list[torch.device]]] = None, use_multi_gpus: bool = False, show_verbose: bool = False, verbose_type: view.VerboseType = view.VerboseType.ALL, **kwargs) -> Module:
        ...

    @overload
    def fit(self, training_dataset: Union[DataLoader[Any], Dataset[Any], Collection], /, epochs: None = None, val_dataset: Optional[Union[DataLoader[Any], Dataset[Any], Collection]] = None, callbacks_list: list[Callback] = [], *, iterations: int, initial_epoch: Optional[int] = None, return_summary: bool = False, device: Optional[Union[torch.device, list[torch.device]]] = None, use_multi_gpus: bool = False, show_verbose: bool = False, verbose_type: view.VerboseType = view.VerboseType.ALL, **kwargs) -> Module:
        ...

    @overload
    def fit(self, training_dataset: Union[DataLoader[Any], Dataset[Any], Collection], /, epochs: int, val_dataset: Optional[Union[DataLoader[Any], Dataset[Any], Collection]] = None, callbacks_list: list[Callback] = [], *, iterations: None = None, initial_epoch: Optional[int] = None, return_summary: bool = True, device: Optional[Union[torch.device, list[torch.device]]] = None, use_multi_gpus: bool = False, show_verbose: bool = False, verbose_type: view.VerboseType = view.VerboseType.ALL, **kwargs) -> tuple[Module, dict[str, float]]:
        ...

    @overload
    def fit(self, training_dataset: Union[DataLoader[Any], Dataset[Any], Collection], /, epochs: None = None, val_dataset: Optional[Union[DataLoader[Any], Dataset[Any], Collection]] = None, callbacks_list: list[Callback] = [], *, iterations: int, initial_epoch: Optional[int] = None, return_summary: bool = True, device: Optional[Union[torch.device, list[torch.device]]] = None, use_multi_gpus: bool = False, show_verbose: bool = False, verbose_type: view.VerboseType = view.VerboseType.ALL, **kwargs) -> tuple[Module, dict[str, float]]:
        ...

    def fit(self, training_dataset: Union[DataLoader[Any], Dataset[Any], Collection], /, epochs: Optional[int] = None, val_dataset: Optional[Union[DataLoader[Any], Dataset[Any], Collection]] = None, callbacks_list: list[Callback] = [], *, iterations: Optional[int] = None, initial_epoch: Optional[int] = None, return_summary: bool = False, device: Optional[Union[torch.device, list[torch.device]]] = None, use_multi_gpus: bool = False, show_verbose: bool = False, verbose_type: view.VerboseType = view.VerboseType.ALL, **kwargs) -> Union[Module, tuple[Module, dict[str, float]]]:
        """
        Training algorithm

        - Parameters:
            - training_dataset: Any kind of training dataset in `torch.utils.data.DataLoader` or `.data.Dataset`
            - epochs: An optional `int` number of training epochs (`iterations` must be not given)
            - val_dataset: An optional validation `Any`
            - callbacks_list: A `list` of callbacks in `Callback`
            - iterations: An optional `int` number of training iterations (`epochs` must be not given)
            - initial_epoch: An optional `int` number of initial epoch
            - return_summary: A `bool` flag of if returning the summary along with the trained model
            - device: An optional `torch.device` to test on if not using multi-GPUs or an optional default `torch.device` for testing otherwise
            - use_multi_gpus: A `bool` flag of if using multi gpus
            - show_verbose: A `bool` flag to show the progress bar during training
            - verbose_type: A `VerboseType` of the summary to show
            - **kwargs: Additional keyword arguments that will be passed to `train` method.
        - Returns: A trained `torch.nn.Module` or a `tuple` of the trained `torch.nn.Module` and a summary of `dict` with keys as `str` and values as `float`
        """
        # arguments checking
        dataset_len = training_dataset.batched_len if isinstance(training_dataset, Dataset) else len(training_dataset)
        assert self.compiled is True, errors._raise(ValueError("Manager has not yet been compiled. Either loss_fn or optimizer, or both, are not given."))

        # check for epochs and iterations
        if epochs is not None:
            assert epochs > 0, errors._raise(ValueError(f"The epochs must be a positive integer, got {epochs}."))
            assert iterations is None, errors._raise(ValueError(f"The iterations must be given as `None` when epochs is given, got {iterations}."))
        else:
            assert iterations is not None, errors._raise(ValueError(f"The iterations must be given if epochs is not given."))
            assert iterations > 0, errors._raise(ValueError(f"The iterations must be a positive integer, got {iterations}."))
            assert epochs is None, errors._raise(ValueError(f"The epochs must be given as `None` when iterations is given, got {epochs}."))
            epochs = math.ceil(iterations / dataset_len)

        # initialize initial epoch
        if initial_epoch is not None:
            assert initial_epoch >= 0, errors._raise(ValueError(f"The initial_epoch must be a non_negative integer, got {initial_epoch}."))
            assert initial_epoch < epochs, errors._raise(ValueError(f"The initial_epoch must be smaller than total epochs, got epochs={epochs} but initial_epoch={initial_epoch}."))
            self.current_epoch = initial_epoch
        elif self.current_epoch > 0:
            initial_epoch = self.current_epoch + 1  # skip the latest current epoch when resuming training
        else:
            initial_epoch = self.current_epoch

        # add progress bar to callbacks
        if show_verbose:
            callbacks_list.append(ProgressBar(dataset_len, verbose_type=verbose_type))

        # find available device
        cpu, device, target_devices = devices.search(device)
        if device.type == cpu or len(target_devices) < 2:
            use_multi_gpus = False
        devices.set_default(target_devices[0])

        # initialize summary
        summary: dict[str, float] = {}

        # on train start
        for callback in callbacks_list:
            callback.on_train_start(initial_epoch)

        try:
            # move to device
            if use_multi_gpus:
                use_multi_gpus = self.data_parallel(target_devices)
            self.to(device)

            # epoch loop
            for self.current_epoch in range(initial_epoch, epochs):
                # calculate batch iterations
                if iterations is not None:
                    batch_iterations = iterations if dataset_len < iterations else iterations
                else:
                    batch_iterations = None

                # calculate iterations per epoch
                iterations_per_epoch = dataset_len if batch_iterations is None else batch_iterations

                # set iterations per epoch in progress bar
                for callback in callbacks_list:
                    if isinstance(callback, ProgressBar):
                        callback.iterations_per_epoch = iterations_per_epoch

                # initialize epoch
                view.logger.info(f"Training epoch {self.current_epoch + 1}/{epochs}")

                # on epoch start
                for callback in callbacks_list:
                    callback.on_epoch_start(self.current_epoch)

                # train for one epoch
                summary = self._train(training_dataset, iterations=batch_iterations, device=device, use_multi_gpus=use_multi_gpus, callbacks_list=callbacks_list, **kwargs)
                if iterations is not None and batch_iterations is not None:
                    iterations -= batch_iterations

                # validate
                val_summary = self.test(val_dataset, device=device, use_multi_gpus=use_multi_gpus, empty_cache=False) if val_dataset is not None else None

                # on epoch end
                for callback in callbacks_list:
                    callback.on_epoch_end(self.current_epoch, summary=summary, val_summary=val_summary)

                # print summary info
                val_message = f"Epoch {self.current_epoch + 1}/{epochs}: "
                if val_summary is not None:
                    summary |= {f"val_{name}": value for name, value in val_summary.items()}
                for i, (name, value) in enumerate(summary.items()):
                    if i > 0:
                        val_message += ", "
                    val_message += f"{name}={value:.4f}"
                view.logger.info(val_message)
        except errors.StopTraining as error:
            pass
        except KeyboardInterrupt:
            view.logger.info("Training interrupted.")
            pass
        except Exception as error:
            view.logger.error(error)
            runtime_error = errors.StopTraining(self.current_epoch, "Training failed.")
            raise runtime_error from error
        finally:
            # on train end
            for callback in callbacks_list:
                callback.on_train_end(self.raw_model)

            # remove added progress bar
            if show_verbose:
                callbacks_list.pop()

            # reset model
            self.reset(cpu)
        return (self.raw_model, summary) if return_summary else self.raw_model

    def train_step(self, x_train: Any, y_train: Any) -> dict[str, float]:
        """
        A single training step

        - Parameters:
            - x_train: The training data
            - y_train: The training label
        - Returns: A summary of `dict` with keys as `str` and values as `float`
        """
        # forward pass
        y, loss = self.forward(x_train, y_train)
        assert loss is not None, _raise(TypeError("Loss cannot be fetched."))

        # forward metrics
        for name, fn in self.compiled_metrics.items():
            if not name.startswith("val_") and "loss" not in name:
                _ = fn(y, y_train)

        # backward pass
        self.compiled_optimizer.zero_grad()
        self.backward(loss)
        self.compiled_optimizer.step()
        return self.summary

    def to_checkpoint(self) -> Checkpoint[Self]:
        ckpt = super().to_checkpoint()
        ckpt.last_epoch = self.current_epoch
        return ckpt
