from __future__ import annotations
from typing import Tuple

from torchmanager_core import devices, gc, math, torch, view
from torchmanager_core.typing import Any, Callable, Dict, Generic, List, Module, Optional, SizedIterable, Type, Union

from .callbacks import Callback
from .losses import Loss, MultiLosses, MultiOutputsLosses
from .metrics import Metric
from .train import Checkpoint, learning_rate

class Manager(Generic[Module]):
    """
    A training manager

    * [Deprecation Warning]: Method `train` becomes protected from v1.0.2, the public method will be removed from v1.2.0. Override `_train` method instead.

    - Properties:
        - compiled_losses: The loss function in `Metric` that must be exist
        - compiled_optimizer: The `torch.optim.Optimizer` that must be exist
        - loss_fn: A `Callable` method that takes the truth and predictions in `torch.Tensor` and returns a loss `torch.Tensor`
        - metrics: A `dict` of metrics with a name in `str` and a `Callable` method that takes the truth and predictions in `torch.Tensor` and returns a loss `torch.Tensor`
        - model: A target `torch.nn.Module` to be trained
        - optimizer: A `torch.optim.Optimizer` to train the model
    """
    # properties
    __compiled: bool
    __current_epoch: int
    loss_fn: Optional[Loss]
    metric_fns: Dict[str, Metric]
    model: Module
    optimizer: Optional[torch.optim.Optimizer]
    
    @property
    def compiled_losses(self) -> Loss:
        assert self.loss_fn is not None, "[Training Error]: loss_fn is not given, compiles the manager with loss_fn first."
        return self.loss_fn

    @property
    def compiled_metrics(self) -> Dict[str, Metric]:
        return {name: m for name, m in self.metric_fns.items() if "loss" not in name}

    @property
    def compiled_optimizer(self) -> torch.optim.Optimizer:
        assert self.optimizer is not None, "[Training Error]: optimizer is not given."
        return self.optimizer

    @property
    def current_epoch(self) -> int:
        return self.__current_epoch

    @current_epoch.setter
    def current_epoch(self, e: int) -> None:
        assert e >= 0, f"[Training Error]: The epoch index must be a non_negative integer, got {e}."
        self.__current_epoch = e
    
    def __init__(self, model: Module, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[Union[Loss, Dict[str, Loss], Callable[[Any, Any], torch.Tensor]]] = None, metrics: Dict[str, Union[Metric, Callable[[Any, Any], torch.Tensor]]] = {}) -> None:
        """
        Constructor
        
        - Parameters:
            - loss_fn: An optional `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: An optional `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
            - model: An optional target `torch.nn.Module` to be trained
            - optimizer: An optional `torch.optim.Optimizer` to train the model
        """
        # initialize
        self.__current_epoch = 0
        self.metric_fns = {}
        self.model = model

        # compile
        self._compile(optimizer, loss_fn, metrics)

        # check compiled
        if self.loss_fn is not None and self.optimizer is not None:
            self.__compiled = True
        else:
            self.__compiled = False

    def _compile(self, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[Union[Loss, Dict[str, Loss], Callable[[Any, Any], torch.Tensor]]] = None, metrics: Dict[str, Union[Metric, Callable[[Any, Any], torch.Tensor]]] = {}) -> None:
        """
        Compiles the manager
        
        - Parameters:
            - loss_fn: An optional `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: An optional `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
            - optimizer: An optional `torch.optim.Optimizer` to train the model
        """
        # initialize loss
        if isinstance(loss_fn, MultiOutputsLosses):
            loss_fn_mapping: Dict[str, Loss] = {f"{name}_loss": fn for name, fn in loss_fn.losses.items()} # type: ignore
            self.metric_fns.update(loss_fn_mapping)
        elif isinstance(loss_fn, dict):
            loss_fn_mapping: Dict[str, Loss] = {f"{name}_loss": fn for name, fn in loss_fn.items()}
            self.metric_fns.update(loss_fn_mapping)
            loss_fn = MultiLosses([l for l in loss_fn_mapping.values()])
        elif loss_fn is not None:
            loss_fn = Loss(loss_fn)
            view.warnings.warn("[Deprecated Warning]: parsing `loss_fn` as a function was deprecated from v0.9.3 and will no longer be available from v1.1.0, use losses.Loss object instead.", DeprecationWarning)
        self.loss_fn = loss_fn

        # initialize metrics
        for name, fn in metrics.items():
            if isinstance(fn, Metric):
                self.metric_fns[name] = fn
            else:
                view.warnings.warn("[Deprecated Warning]: parsing a metric in `metrics` as a function was deprecated from v0.9.3 and will no longer be available from v1.1.0, use `metrics.Metric` object instead.", DeprecationWarning)
                self.metric_fns[name] = Metric(fn)

        # initialize optimizer
        self.optimizer = optimizer

    def _train(self, dataset: SizedIterable, iterations: Optional[int] = None, device: torch.device = devices.CPU, use_multi_gpus: bool = False, show_verbose: bool = False, verbose_type: view.VerboseType = view.VerboseType.ALL, callbacks_list: List[Callback] = []) -> Dict[str, float]:
        """
        The single training step for an epoch

        - Parameters:
            - dataset: A `SizedIterable` training dataset
            - device: A `torch.device` where the data is moved to, should be same as the model
            - use_multi_gpus: A `bool` flag of if using multi gpus
            - show_verbose: A `bool` flag of if showing progress bar
            - callbacks_list: A `list` of callbacks in `Callback`
        - Returns: A summary of `dict` with keys as `str` and values as `float`
        """
        # reset loss and metrics
        self.compiled_losses.reset()
        for _, m in self.metric_fns.items(): m.reset()

        # run deprecated method
        summary = self.train(dataset, device=device, use_multi_gpus=use_multi_gpus, show_verbose=show_verbose, callbacks_list=callbacks_list)
        if summary != NotImplemented:
            view.warnings.warn("[Deprecated Warning]: Method `train` will be set to private in a future version. Override `_train` instead.", PendingDeprecationWarning)
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
            x_train, y_train = self._unpack_data(data)
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
                else: progress_summary = None

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
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return summary

    def _unpack_data(self, data: Any) -> Tuple[Any, Any]:
        """Unpacks data"""
        (x_train, y_train) = data
        return x_train, y_train

    def compile(self, optimizer: torch.optim.Optimizer, loss_fn: Union[Loss, Dict[str, Loss], Callable[[Any, Any], torch.Tensor]], metrics: Dict[str, Union[Metric, Callable[[Any, Any], torch.Tensor]]] = {}) -> None:
        """
        Compiles the manager
        
        - Parameters:
            - loss_fn: A `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: A `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
            - optimizer: A `torch.optim.Optimizer` to train the model
        """
        self._compile(optimizer, loss_fn, metrics)
        self.__compiled = True

    def fit(self, training_dataset: Any, epochs: Optional[int] = None, iterations: Optional[int] = None, initial_epoch: Optional[int] = None, lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, val_dataset: Optional[Any] = None, device: Optional[torch.device] = None, use_multi_gpus: bool = False, callbacks_list: List[Callback] = [], **kwargs) -> torch.nn.Module:
        """
        Training algorithm

        - Parameters:
            - training_dataset: Any kind of training dataset, must performs to `SizedIterable`
            - epochs: An optional `int` number of training epochs
            - iterations: An optional `int` number of training iterations
            - lr_scheduelr: An optioanl `torch.optim.lr_scheduler._LRScheduler` to update the lr per epoch
            - is_dynamic_pruning: A `bool` flag of if using dynamic pruning
            - val_dataset: An optional validation `Any`
            - device: An optional `torch.device` where the data is moved to, gpu will be used when available if not specified.
            - use_multi_gpus: A `bool` flag of if using multi gpus
            - callbacks_list: A `list` of callbacks in `Callback`
            - **kwargs: Additional keyword arguments that will be passed to `train` method.
        - Returns: A trained `torch.nn.Module`
        """
        # arguments checking
        assert self.__compiled is True, "[Training Error]: Manager has not yet been compiled. Either loss_fn or optimizer, or both, are not given."
        assert isinstance(training_dataset, SizedIterable), "[Runtime Error]: The training_dataset must be both Sized and Iterable."
        if epochs is not None:
            assert epochs > 0, f"[Training Error]: The epochs must be a positive integer, got {epochs}."
            assert iterations is None, f"[Training Error]: The iterations must be given as `None` when epochs is given, got {iterations}."
            iterations = int(epochs * len(training_dataset))
        else:
            assert iterations is not None, f"[Training Error]: The iterations must be given if epochs is not given."
            assert iterations > 0, f"[Training Error]: The iterations must be a positive integer, got {iterations}."
            assert epochs is None, f"[Training Error]: The epochs must be given as `None` when iterations is given, got {epochs}."
            epochs = math.ceil(iterations / len(training_dataset))

        # initialize initial epoch
        if initial_epoch is not None: 
            assert initial_epoch >= 0, f"[Training Error]: The initial_epoch must be a non_negative integer, got {initial_epoch}."
            assert initial_epoch < epochs, f"[Training Error]: The initial_epoch must be smaller than total epochs, got epochs={epochs} but initial_epoch={initial_epoch}."
            self.__current_epoch = initial_epoch
        else: initial_epoch = 0

        # initialize
        view.logging.basicConfig(level=view.logging.INFO, format="%(message)s")
        logger = view.logging.getLogger("Training")
        learning_rate.initial_step_lr_scheduler(lr_scheduler, self.__current_epoch)
        cpu, device = devices.find(device)
        for c in callbacks_list: c.on_train_start(initial_epoch)
        
        # multi gpus support
        raw_model = self.model
        raw_loss_fn = self.compiled_losses
        if use_multi_gpus is True: 
            self.model, use_multi_gpus = devices.data_parallel(raw_model)
            paralleled_loss_fn, _ = devices.data_parallel(self.compiled_losses)
            self.loss_fn = Loss(paralleled_loss_fn)
        devices.move_to_device([self.model, self.compiled_losses, self.metric_fns], device)

        # epoch loop
        for epoch in range(self.current_epoch, epochs):
            # initialize epoch
            logger.info(f"Training epoch {epoch + 1}/{epochs}")
            self.model.train()
            for c in callbacks_list: c.on_epoch_start(epoch)
            batch_iterations = len(training_dataset) if len(training_dataset) < iterations else iterations

            # train for one epoch
            summary = self._train(training_dataset, iterations=batch_iterations, device=device, use_multi_gpus=use_multi_gpus, callbacks_list=callbacks_list, **kwargs)
            iterations -= batch_iterations

            # validate
            val_summary = self.test(val_dataset, use_multi_gpus=use_multi_gpus) if val_dataset is not None else {}

            # step lr scheduler
            if lr_scheduler is not None:
                lr_summary = learning_rate.update_lr(lr_scheduler)
                summary.update(lr_summary)

            # on epoch end
            for c in callbacks_list:
                c.on_epoch_end(epoch, summary=summary, val_summary=val_summary)
            self.current_epoch += 1

            # print summary info
            val_message = f"Epoch {epoch + 1}/{epochs}: "
            summary.update({f"val_{name}": value for name, value in val_summary.items()})
            for i, (name, value) in enumerate(summary.items()):
                if i > 0: val_message += ", "
                val_message += f"{name}={value:.4f}"
            logger.info(val_message)

        # remove model from gpu
        self.model = raw_model.to(cpu)
        self.loss_fn = raw_loss_fn
        devices.empty_cache()
        return self.model

    def train(self, *args, **kwargs) -> Dict[str, float]:
        """The single training step for an epoch"""
        return NotImplemented

    @classmethod
    def from_checkpoint(cls: Type[Manager[torch.nn.Module]], *args, **kwargs) -> Manager[torch.nn.Module]:
        """
        Method to load a manager from a saved `Checkpoint`. The manager will not be compiled with a loss function and its metrics.

        - Returns: A loaded `Manager`
        """
        # load checkpoint
        ckpt = Checkpoint.from_saved(*args, **kwargs)

        # recover model to manager
        if isinstance(ckpt.model, torch.nn.Module):
            manager = cls(ckpt.model, ckpt.optimizer, loss_fn=ckpt.loss_fn, metrics=ckpt.metrics)
            manager.current_epoch = ckpt.last_epoch
            return manager
        elif isinstance(ckpt.model, Manager):
            return ckpt.model
        else: raise TypeError(f"[Ckpt Error]: The saved checkpoint contains a model with type of {type(ckpt.model)} that cannot be recoverred to a `Manager`.")

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
        self.compiled_optimizer.zero_grad()
        y = self.model(x_train)
        loss = self.compiled_losses(y, y_train)

        # forward metrics
        for name, fn in self.compiled_metrics.items():
            if not name.startswith("val_"): fn(y, y_train)

        # backward pass
        loss.backward()
        self.compiled_optimizer.step()

        # summary result
        try: summary["loss"] = float(self.compiled_losses.result.detach())
        except Exception as e: raise RuntimeError("[Runtime Error]: Cannot fetch loss.") from e
        for name, fn in self.metric_fns.items():
            if name.startswith("val_"): continue
            try: summary[name] = float(fn.result.detach())
            except Exception as metric_error:
                runtime_error = RuntimeError(f"[Runtime Error]: Cannot fetch metric '{name}'.")
                raise runtime_error from metric_error

        return summary

    def test(self, dataset: Any, device: Optional[torch.device] = None, use_multi_gpus: bool = False, show_verbose: bool = False) -> Dict[str, float]:
        """
        Test target model

        - Parameters:
            - dataset: Either `SizedIterable` or `data.DataLoader` to load the dataset
            - use_multi_gpus: A `bool` flag to use multi gpus during testing
        - Returns: A `dict` of validation summary
        """
        # arguments checking
        assert isinstance(dataset, SizedIterable), "[Runtime Error]: The dataset must be both Sized and Iterable."

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

            if not isinstance(self.compiled_losses, torch.nn.parallel.DataParallel):
                raw_loss_fn = self.compiled_losses
                paralleled_loss_fn = torch.nn.parallel.DataParallel(self.compiled_losses)
                self.loss_fn = Loss(paralleled_loss_fn)
            else: raw_loss_fn = None
        else:
            raw_model = None
            raw_loss_fn = None

        # set module status
        try:
            self.model.eval()
            devices.move_to_device([self.model, self.loss_fn, self.metric_fns], device)
        except: pass

        # initialize progress bar
        if len(dataset) == 0: return {}
        progress_bar = view.tqdm(total=len(dataset)) if show_verbose else None

        # disable auto gradients
        with torch.no_grad():
            # batch loop
            for data in dataset:
                # move x_test, y_test to device
                x_test, y_test = self._unpack_data(data)
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
                if name.startswith("val_"): name.replace("val_", "")
                try: summary[name] = float(fn.result.detach())
                except Exception as metric_error:
                    runtime_error = RuntimeError(f"Cannot fetrch metric '{name}'.")
                    raise runtime_error from metric_error
            if self.loss_fn is not None:
                summary["loss"] = float(self.compiled_losses.result.detach())

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
            try:
                fn(y, y_test)
                summary[name] = float(fn.result.detach())
            except Exception as metric_error:
                runtime_error = RuntimeError(f"[Runtime Error]: Cannot fetch metric '{name}'.")
                raise runtime_error from metric_error

        # forward loss
        if self.loss_fn is not None:
            self.loss_fn(y, y_test)
            summary["loss"] = float(self.loss_fn.result.detach())
        return summary

    def to_checkpoint(self) -> Checkpoint[Module]:
        """
        Convert the current manager to a checkpoint
        
        - Returns: A `Checkpoint` with its model in `Module` type
        """
        metrics: Dict[str, torch.nn.Module] = {k: m for k, m in self.metric_fns.items()}
        ckpt = Checkpoint(self.model, last_epoch=self.current_epoch, optimizer=self.optimizer, loss_fn=self.loss_fn, metrics=metrics)
        return ckpt