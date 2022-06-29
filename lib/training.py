from torchmanager_core import devices, math, torch, view
from torchmanager_core.typing import Any, Callable, Dict, Generic, List, Module, Optional, SizedIterable, Union

from .callbacks import Callback
from .losses import Loss
from .metrics import Metric
from .train import Checkpoint, learning_rate
from .testing import Manager as _Manager

class Manager(_Manager, Generic[Module]):
    """
    A training manager

    * extends: `.testing.Manager`
    * [Deprecation Warning]: Method `train` becomes protected from v1.0.2, the public method will be removed from v1.2.0. Override `_train` method instead.

    - Properties:
        - current_epoch: The `int` index of current training epoch
        - compiled_optimizer: The `torch.optim.Optimizer` that must be exist
    """
    __current_epoch: int
    model: Module

    @property
    def current_epoch(self) -> int:
        return self.__current_epoch

    @current_epoch.setter
    def current_epoch(self, e: int) -> None:
        assert e >= 0, f"[Training Error]: The epoch index must be a non_negative integer, got {e}."
        self.__current_epoch = e
    
    @property
    def compiled_losses(self) -> Loss:
        assert self.loss_fn is not None, "[Training Error]: loss_fn is not given, compiles the manager with loss_fn first."
        return self.loss_fn

    @property
    def compiled_optimizer(self) -> torch.optim.Optimizer:
        assert self.optimizer is not None, "[Training Error]: optimizer is not given."
        return self.optimizer

    def __init__(self, model: Module, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[Union[Loss, Dict[str, Loss], Callable[[Any, Any], torch.Tensor]]] = None, metrics: Dict[str, Union[Metric, Callable[[Any, Any], torch.Tensor]]] = ...) -> None:
        super().__init__(model, optimizer, loss_fn, metrics)
        self.__current_epoch = 0

    def _train(self, dataset: SizedIterable, iterations: Optional[int] = None, device: torch.device = devices.CPU, use_multi_gpus: bool = False, show_verbose: bool = False, verbose_type: view.VerboseType = view.VerboseType.ALL, callbacks_list: List[Callback] = []) -> Dict[str, float]:
        """
        The single training step for an epoch

        - Parameters:
            - dataset: A `SizedIterable` training dataset
            - iterations: An optional `int` of total training iterations, must be smaller than the size of dataset
            - device: A `torch.device` where the data is moved to, should be same as the model
            - use_multi_gpus: A `bool` flag of if using multi gpus
            - show_verbose: A `bool` flag of if showing progress bar
            - verbose_type: A `view.VerboseType` that controls the display of verbose
            - callbacks_list: A `list` of callbacks in `Callback`
        - Returns: A summary of `dict` with keys as `str` and values as `float`
        """
        # reset loss and metrics
        self.compiled_losses.reset()
        for _, m in self.metric_fns.items(): m.reset()

        # run deprecated method
        summary = self.train(dataset, device=device, use_multi_gpus=use_multi_gpus, show_verbose=show_verbose, callbacks_list=callbacks_list)
        if summary is not NotImplemented:
            view.warnings.warn("[Deprecation Warning]: Method `train` will be set to protected in v1.1.0 and will be removed in v1.2.0, override `_train` instead.", PendingDeprecationWarning)
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
                else: raise TypeError(f"[Runtime Error]: Verbose type {verbose_type} is not supported.")

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
        devices.empty_cache()
        return summary

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
        assert self.compiled is True, "[Training Error]: Manager has not yet been compiled. Either loss_fn or optimizer, or both, are not given."
        assert isinstance(training_dataset, SizedIterable), "[Runtime Error]: The training_dataset must be both Sized and Iterable."
        if epochs is not None:
            assert epochs > 0, f"[Training Error]: The epochs must be a positive integer, got {epochs}."
            assert iterations is None, f"[Training Error]: The iterations must be given as `None` when epochs is given, got {iterations}."
        else:
            assert iterations is not None, f"[Training Error]: The iterations must be given if epochs is not given."
            assert iterations > 0, f"[Training Error]: The iterations must be a positive integer, got {iterations}."
            assert epochs is None, f"[Training Error]: The epochs must be given as `None` when iterations is given, got {epochs}."
            epochs = math.ceil(iterations / len(training_dataset))

        # initialize initial epoch
        if initial_epoch is not None: 
            assert initial_epoch >= 0, f"[Training Error]: The initial_epoch must be a non_negative integer, got {initial_epoch}."
            assert initial_epoch < epochs, f"[Training Error]: The initial_epoch must be smaller than total epochs, got epochs={epochs} but initial_epoch={initial_epoch}."
            self.current_epoch = initial_epoch
        else: initial_epoch = self.current_epoch

        # initialize
        view.logging.basicConfig(level=view.logging.INFO, format="%(message)s")
        logger = view.logging.getLogger("Training")
        learning_rate.initial_step_lr_scheduler(lr_scheduler, self.current_epoch)
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
        for self.current_epoch in range(initial_epoch, epochs):
            # initialize epoch
            logger.info(f"Training epoch {self.current_epoch + 1}/{epochs}")
            self.model.train()
            for c in callbacks_list: c.on_epoch_start(self.current_epoch)
            if iterations is not None: batch_iterations = iterations if len(training_dataset) < iterations else iterations
            else: batch_iterations = None

            # train for one epoch
            summary = self._train(training_dataset, iterations=batch_iterations, device=device, use_multi_gpus=use_multi_gpus, callbacks_list=callbacks_list, **kwargs)
            if iterations is not None and batch_iterations is not None: iterations -= batch_iterations

            # validate
            val_summary = self.test(val_dataset, device=device, use_multi_gpus=use_multi_gpus) if val_dataset is not None else {}

            # on epoch end
            for c in callbacks_list:
                c.on_epoch_end(self.current_epoch, summary=summary, val_summary=val_summary)

            # step lr scheduler
            if lr_scheduler is not None:
                lr_summary = learning_rate.update_lr(lr_scheduler)
                summary.update(lr_summary)

            # print summary info
            val_message = f"Epoch {self.current_epoch + 1}/{epochs}: "
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

    def to_checkpoint(self) -> Checkpoint[Module]:
        ckpt = super().to_checkpoint()
        ckpt.last_epoch = self.current_epoch
        return ckpt