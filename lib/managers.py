# import typing modules
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Protocol, Type, Union, runtime_checkable

# import required modules
import abc, torch, warnings
from torch.utils import data
from tqdm import tqdm

# import core modules
from .callbacks import Callback
from .train import Checkpoint
from .losses import Loss, MultiLosses
from .metrics import Metric

@runtime_checkable
class _VerboseControllable(Protocol):
    '''
    The learning rate scheduler protocol

    - Properties:
        - verbose: A `bool` flag of if showing messages when updating lr
    '''
    @abc.abstractproperty
    def verbose(self) -> bool:
        raise NotImplementedError

    @verbose.setter
    @abc.abstractmethod
    def verbose(self, verbose: bool) -> None:
        raise NotImplementedError

class Manager:
    '''
    A training manager

    - Properties:
        - compiled_losses: The loss function in `Metric` that must be exist
        - compiled_optimizer: The `torch.optim.Optimizer` that must be exist
        - loss_fn: A `Callable` method that takes the truth and predictions in `torch.Tensor` and returns a loss `torch.Tensor`
        - metrics: A `dict` of metrics with a name in `str` and a `Callable` method that takes the truth and predictions in `torch.Tensor` and returns a loss `torch.Tensor`
        - model: A target `torch.nn.Module` to be trained
        - optimizer: A `torch.optim.Optimizer` to train the model
    '''
    # properties
    __compiled: bool
    loss_fn: Optional[Union[Loss, Dict[str, Loss]]]
    metric_fns: Dict[str, Metric]
    model: torch.nn.Module
    optimizer: Optional[torch.optim.Optimizer]
    
    @property
    def compiled_losses(self) -> Loss:
        assert self.loss_fn is not None, "[Training Error]: loss_fn is not given, compiles the manager with loss_fn first."
        if isinstance(self.loss_fn, dict):
            loss_fn = {f"loss_{name}": fn for name, fn in self.loss_fn.items()}
            self.metric_fns.update(loss_fn)
            self.loss_fn = MultiLosses([l for l in self.loss_fn.values()])
        return self.loss_fn

    @property
    def compiled_metrics(self) -> Dict[str, Metric]:
        return {name: m for name, m in self.metric_fns.items() if "loss" not in name}

    @property
    def compiled_optimizer(self) -> torch.optim.Optimizer:
        assert self.optimizer is not None, "[Training Error]: optimizer is not given."
        return self.optimizer
    
    def __init__(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer]=None, loss_fn: Optional[Union[Loss, Dict[str, Loss], Callable[[Any, Any], torch.Tensor]]]=None, metrics: Dict[str, Union[Metric, Callable[[Any, Any], torch.Tensor]]]={}) -> None:
        '''
        Constructor
        
        - Parameters:
            - loss_fn: An optional `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: An optional `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
            - model: An optional target `torch.nn.Module` to be trained
            - optimizer: An optional `torch.optim.Optimizer` to train the model
        '''
        # initialize loss
        if isinstance(loss_fn, Loss) or isinstance(loss_fn, dict):
            self.loss_fn = loss_fn 
        elif loss_fn is not None:
            self.loss_fn = Loss(loss_fn)
            warnings.warn("[Deprecated Warning]: parsing `loss_fn` as a function was deprecated from v0.9.3 and will no longer be available from v1.1.0, use losses.Loss object instead.", DeprecationWarning)
        else:
            self.loss_fn = None

        # initialize metrics
        self.metric_fns = {}
        for name, fn in metrics.items():
            if isinstance(fn, Metric):
                self.metric_fns[name] = fn
            else:
                warnings.warn("[Deprecated Warning]: parsing a metric in `metrics` as a function was deprecated from v0.9.3 and will no longer be available from v1.1.0, use `metrics.Metric` object instead.", DeprecationWarning)
                self.metric_fns[name] = Metric(fn)

        # initialize main model and optimizer
        self.model = model
        self.optimizer = optimizer

        # check compiled
        if self.loss_fn is not None and self.optimizer is not None:
            self.__compiled = True
        else:
            self.__compiled = False

    def __compile(self, optimizer: Optional[torch.optim.Optimizer]=None, loss_fn: Optional[Union[Loss, Dict[str, Loss], Callable[[Any, Any], torch.Tensor]]]=None, metrics: Dict[str, Union[Metric, Callable[[Any, Any], torch.Tensor]]]={}) -> None:
        '''
        Compiles the manager
        
        - Parameters:
            - loss_fn: An optional `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: An optional `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
            - optimizer: An optional `torch.optim.Optimizer` to train the model
        '''
        # initialize loss
        if isinstance(loss_fn, Loss) or isinstance(loss_fn, dict):
            self.loss_fn = loss_fn 
        elif loss_fn is not None:
            self.loss_fn = Loss(loss_fn)
            warnings.warn("[Deprecated Warning]: parsing `loss_fn` as a function was deprecated from v0.9.3 and will no longer be available from v1.1.0, use losses.Loss object instead.", DeprecationWarning)
        else:
            self.loss_fn = None

        # initialize metrics
        self.metric_fns = {}
        for name, fn in metrics.items():
            if isinstance(fn, Metric):
                self.metric_fns[name] = fn
            else:
                warnings.warn("[Deprecated Warning]: parsing a metric in `metrics` as a function was deprecated from v0.9.3 and will no longer be available from v1.1.0, use `metrics.Metric` object instead.", DeprecationWarning)
                self.metric_fns[name] = Metric(fn)

        # initialize optimizer
        self.optimizer = optimizer

    def compile(self, optimizer: torch.optim.Optimizer, loss_fn: Union[Loss, Dict[str, Loss], Callable[[Any, Any], torch.Tensor]], metrics: Dict[str, Union[Metric, Callable[[Any, Any], torch.Tensor]]]={}) -> None:
        '''
        Compiles the manager
        
        - Parameters:
            - loss_fn: A `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: A `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
            - optimizer: A `torch.optim.Optimizer` to train the model
        '''
        self.__compile(optimizer, loss_fn, metrics)
        self.__compiled = True

    def fit(self, training_dataset: data.DataLoader, epochs: int=100, initial_epoch: int=0, lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]=None, show_verbose: bool=False, val_dataset: Optional[data.DataLoader]=None, device: Optional[torch.device]=None, use_multi_gpus: bool=False, callbacks_list: List[Callback]=[], **kwargs) -> torch.nn.Module:
        '''
        Training algorithm

        - Parameters:
            - training_dataset: The `data.DataLoader` for training dataset
            - epochs: The `int` number of training epochs
            - lr_scheduelr: An optioanl `torch.optim.lr_scheduler._LRScheduler` to update the lr per epoch
            - is_dynamic_pruning: A `bool` flag of if using dynamic pruning
            - show_verbose: A `bool` flag of if showing progress bar
            - val_dataset: An optional validation `data.DataLoader`
            - device: An optional `torch.device` where the data is moved to, gpu will be used when available if not specified.
            - use_multi_gpus: A `bool` flag of if using multi gpus
            - callbacks_list: A `list` of callbacks in `Callback`
            - **kwargs: Additional keyword arguments that will be passed to `train_step` method. If given, `train` method must be overriden to accept these arguments.
        - Returns: A trained `torch.nn.Module`
        '''
        # ensure compiled and epochs
        assert self.__compiled is True, "[Training Error]: Manager has not yet been compiled. Either loss_fn or optimizer, or both, are not given."
        assert epochs > 0, f"[Training Error]: The epochs must be a positive integer, got {epochs}."
        assert initial_epoch >= 0, f"[Training Error]: The initial_epoch must be a non_negative integer, got {initial_epoch}."
        assert initial_epoch < epochs, f"[Training Error]: The initial_epoch must be smaller than total epochs, got epochs={epochs} but initial_epoch={initial_epoch}."

        # initialize device
        cpu = torch.device("cpu")
        if device is None:
            gpu = torch.device("cuda")
            device = gpu if torch.cuda.is_available() else cpu
        else:
            warnings.warn(f"[Device Warning]: Using specified device {device}.", ResourceWarning)
        
        # multi gpus support
        raw_model = self.model
        if use_multi_gpus is True:
            if torch.cuda.is_available():
                self.model = torch.nn.parallel.DataParallel(raw_model)
            else:
                use_multi_gpus = False
                warnings.warn(f"[Device Warning]: The use_multi_gpus flag is set to True, but CUDA is not available.", ResourceWarning)
        self.model.to(device)

        # move losses to device
        if isinstance(self.loss_fn, dict):
            for l in self.loss_fn.values(): l.to(device)
        elif isinstance(self.loss_fn, MultiLosses):
            for l in self.loss_fn.losses: l.to(device)
        else:
            self.compiled_losses.to(device)

        # move metrics to device
        for m in self.metric_fns.values(): m.to(device)

        # on train start
        for c in callbacks_list:
            c.on_train_start()

        # go to initial epoch
        if lr_scheduler is not None and initial_epoch > 0:
            # disable verbose
            assert isinstance(lr_scheduler, _VerboseControllable), "[Runtime Error]: lr_scheduler does not performs to the VerboseControllable protocol."
            verbose = lr_scheduler.verbose
            lr_scheduler.verbose = False

            # steps to initial epoch
            for _ in range(initial_epoch):
                lr_scheduler.step()

            # reset verbose
            lr_scheduler.verbose = verbose

        # epoch loop
        for epoch in range(initial_epoch, epochs):
            # initialize epoch
            print(f"Training epoch {epoch + 1}/{epochs}")

            # on epoch start
            for c in callbacks_list:
                c.on_epoch_start(epoch)

            # train for one epoch
            summary = self.train(training_dataset, device=device, use_multi_gpus=use_multi_gpus, show_verbose=show_verbose, callbacks_list=callbacks_list, **kwargs)

            # step lr scheduler
            if lr_scheduler is not None:
                lr_scheduler.step()

            # validate
            if val_dataset is not None:
                val_summary = self.test(val_dataset, use_multi_gpus=use_multi_gpus)

                # print epoch info
                print(f"Epoch {epoch + 1}/{epochs}:", end=" ")

                # print summary info
                for name, value in summary.items():
                    print(f"{name}={value:.4f}", end=", ")
                
                # print val summary info
                for i, (name, value) in enumerate(val_summary.items()):
                    if i > 0: print(", ", end="")
                    print(f"val_{name}={value:.4f}", end="")
                print()
            else:
                val_summary = None

                # print epoch info
                print(f"Epoch {epoch + 1}/{epochs}:", end=" ")

                # print summary info
                for i, (name, value) in enumerate(summary.items()):
                    if i > 0: print(", ")
                    print(f"{name}={value:.4f}", end="")
                print()

            # on epoch end
            for c in callbacks_list:
                c.on_epoch_end(epoch, summary=summary, val_summary=val_summary)

        # remove model from gpu
        self.model = raw_model.to(cpu)
        return self.model

    @classmethod
    def from_checkpoint(cls: Type[Manager], *args, **kwargs) -> Manager:
        '''
        Method to load a manager from a saved `Checkpoint`. The manager will not be compiled with a loss function and its metrics.

        - Returns: A loaded `Manager`
        '''
        # load checkpoint
        ckpt = Checkpoint.from_saved(*args, **kwargs)
        return cls(ckpt.model, ckpt.optimizer, loss_fn=ckpt.loss_fn, metrics=ckpt.metrics)

    def train(self, dataset: data.DataLoader, device: torch.device=torch.device('cpu'), use_multi_gpus: bool=False, show_verbose: bool=False, callbacks_list: List[Callback]=[]) -> Dict[str, float]:
        '''
        The single training step for an epoch

        - Parameters:
            - dataset: The `data.DataLoader` for training dataset
            - device: A `torch.device` where the data is moved to, should be same as the model
            - use_multi_gpus: A `bool` flag of if using multi gpus
            - show_verbose: A `bool` flag of if showing progress bar
            - callbacks_list: A `list` of callbacks in `Callback`
        '''
        # initialize
        self.compiled_losses.reset()
        for _, m in self.metric_fns.items(): m.reset()
        self.model.train()

        # initialize progress bar
        if show_verbose is True:
            progress_bar = tqdm(total=len(dataset))
        else:
            progress_bar = None

        # batch loop
        for batch, (x_train, y_train) in enumerate(dataset):
            # on batch start
            for c in callbacks_list:
                c.on_batch_start(batch)

            # move x_train to gpu
            if use_multi_gpus is not True and isinstance(x_train, torch.Tensor):
                x_train = x_train.to(device)

            # move y_train to gpu
            if isinstance(y_train, torch.Tensor):
                y_train = y_train.to(device)

            # train for one step
            summary = self.train_step(x_train, y_train)

            # on batch start
            for c in callbacks_list:
                c.on_batch_end(batch, summary=summary)

            # implement progress bar
            if progress_bar is not None:
                progress_bar.set_postfix(summary)
                progress_bar.update()

        # end epoch training
        if progress_bar is not None:
            progress_bar.close()

        # summarize
        summary = {name: float(fn.result.detach()) for name, fn in self.metric_fns.items()}
        summary["loss"] = float(self.compiled_losses.result.detach())
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return summary

    def train_step(self, x_train: Any, y_train: Any) -> Dict[str, float]:
        '''
        A single training step

        - Parameters:
            - x_train: The training data
            - y_train: The training label
        - Returns: A summary of `dict` with keys as `str` and values as `float`
        '''
        # forward pass
        summary: dict[str, float] = {}
        self.compiled_optimizer.zero_grad()
        y = self.model(x_train)
        loss = self.compiled_losses(y, y_train)

        # forward metrics
        for name, fn in self.compiled_metrics.items():
            fn(y, y_train)

        # backward pass
        loss.backward()
        self.compiled_optimizer.step()

        # summary result
        summary["loss"] = float(self.compiled_losses.result.detach())
        for name, fn in self.metric_fns.items():
            summary[name] = float(fn.result.detach())
        return summary

    def test(self, dataset: data.DataLoader, device: Optional[torch.device]=None, use_multi_gpus: bool=False, show_verbose: bool=False) -> Dict[str, float]:
        '''
        Test target model

        - Parameters:
            - dataset: A `data.DataLoader` to load the dataset
            - use_multi_gpus: A `bool` flag to use multi gpus during testing
        - Returns: A `dict` of validation summary
        '''
        # initialize function
        self.compiled_losses.reset()
        for _, m in self.metric_fns.items(): m.reset()

        # find available device
        cpu = torch.device("cpu")
        if device is None:
            gpu = torch.device("cuda")
            device = gpu if torch.cuda.is_available() else cpu
            use_multi_gpus = torch.cuda.is_available() if use_multi_gpus is True else use_multi_gpus
        else:
            warnings.warn(f"[Device Warning]: Using specified device {device}.", ResourceWarning)

        # multi gpu support
        if use_multi_gpus is True and not isinstance(self.model, torch.nn.parallel.DataParallel):
            raw_model = self.model
            self.model = torch.nn.parallel.DataParallel(self.model)
        else: raw_model = None

        # set module status
        try:
            self.model.eval()
            self.model.to(device)
        except: pass

        # initialize progress bar
        if show_verbose is True:
            progress_bar = tqdm(total=len(dataset))
        else:
            progress_bar = None

        # disable auto gradients
        with torch.no_grad():
            # batch loop
            for x_test, y_test in dataset:
                # move x_train to device
                if use_multi_gpus is not True and isinstance(x_test, torch.Tensor):
                    x_test = x_test.to(device)

                # move y_test to gpu
                if isinstance(y_test, torch.Tensor):
                    y_test = y_test.to(device)

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
            summary: dict[str, float] = {}
            for name, fn in self.metric_fns.items():
                try: summary[name] = float(fn.result.detach())
                except Exception as metric_error:
                    runtime_error = RuntimeError(f"Cannot fetrch metric '{name}'.")
                    raise runtime_error from metric_error
            if self.loss_fn is not None:
                summary["loss"] = float(self.compiled_losses.result.detach())

            # reset model
            if raw_model is not None:
                self.model = raw_model.to(cpu)
            return summary

    def test_step(self, x_test: Any, y_test: Any) -> dict[str, float]:
        '''
        A single testing step

        - Parameters:
            - x_train: The testing data in `torch.Tensor`
            - y_train: The testing label in `torch.Tensor`
        '''
        # initialize
        summary: dict[str, float] = {}

        # forward pass
        y = self.model(x_test)

        # forward metrics
        for name, fn in self.compiled_metrics.items():
            try:
                fn(y, y_test)
                summary[name] = float(fn.result.detach())
            except Exception as metric_error:
                runtime_error = RuntimeError(f"Cannot fetch metric '{name}'.")
                raise runtime_error from metric_error

        # forward loss
        if self.loss_fn is not None:
            self.compiled_losses(y, y_test)
            summary["loss"] = float(self.compiled_losses.result.detach())
        return summary