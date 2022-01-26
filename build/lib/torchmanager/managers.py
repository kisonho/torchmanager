# import typing modules
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Type, Union

# import required modules
import torch, warnings
from torch.utils import data
from tqdm import tqdm

# import core modules
from .callbacks import Callback, Checkpoint
from .losses import Loss, MultiLosses
from .metrics import Metric

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
    __compiled: bool = False
    loss_fn: Optional[Union[Loss, Dict[str, Loss]]] = None
    metrics: Dict[str, Metric] = {}
    model: torch.nn.Module
    optimizer: Optional[torch.optim.Optimizer] = None

    @property
    def compiled_losses(self) -> Loss:
        assert self.loss_fn is not None, "[Training Error]: loss_fn is not given"
        if isinstance(self.loss_fn, dict):
            assert "loss" not in self.loss_fn, "[Loss Error]: Name \'loss\' must not be given in a dictionary of loss_fn."
            self.metrics.update(self.loss_fn)
            return MultiLosses([l for l in self.loss_fn.values()])
        else:
            return self.loss_fn

    @property
    def compiled_optimizer(self) -> torch.optim.Optimizer:
        assert self.optimizer is not None, "[Training Error]: optimizer is not given."
        return self.optimizer
    
    def __init__(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer]=None, loss_fn: Optional[Union[Loss, Dict[str, Loss], Callable[[Any, Any], torch.Tensor]]]=None, metrics: Dict[str, Union[Metric, Callable[[Any, Any], torch.Tensor]]]={}) -> None:
        '''
        Constructor
        
        - Parameters:
            - loss_fn: A `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: A `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
            - model: A target `torch.nn.Module` to be trained
            - optimizer: A `torch.optim.Optimizer` to train the model
        '''
        # initialize loss
        if isinstance(loss_fn, Loss) or isinstance(loss_fn, dict):
            self.loss_fn = loss_fn 
        elif loss_fn is not None:
            self.loss_fn = Loss(loss_fn)
            warnings.warn("[Deprecated Warning]: parsing `loss_fn` as a function was deprecated from v0.9.3, use losses.Loss object instead.", PendingDeprecationWarning)

        # initialize metrics
        for name, fn in metrics.items():
            if isinstance(fn, Metric):
                self.metrics[name] = fn
            else:
                warnings.warn("[Deprecated Warning]: parsing a metric in `metrics` as a function was deprecated from v0.9.3, use `metrics.Metric` object instead.", PendingDeprecationWarning)
                self.metrics[name] = Metric(fn)

        # initialize main model and optimizer
        self.model = model
        self.optimizer = optimizer

        # check compiled
        if self.loss_fn is not None and self.optimizer is not None:
            self.__compiled = True

    def fit(self, training_dataset: data.DataLoader, epochs: int=100, lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]=None, show_verbose: bool=False, val_dataset: Optional[data.DataLoader]=None, use_multi_gpus: bool=False, callbacks_list: List[Callback]=[], **kwargs) -> torch.nn.Module:
        '''
        Training algorithm

        - Parameters:
            - training_dataset: The `data.DataLoader` for training dataset
            - epochs: The `int` number of training epochs
            - lr_scheduelr: An optioanl `torch.optim.lr_scheduler._LRScheduler` for optimizer
            - is_dynamic_pruning: A `bool` flag of if using dynamic pruning
            - show_verbose: A `bool` flag of if showing progress bar
            - val_dataset: An optional validation `data.DataLoader`
            - use_multi_gpus: A `bool` flag of if using multi gpus
            - callbacks_list: A `list` of callbacks in `Callback`
            - **kwargs: Additional keyword arguments that will be passed to `train_step` method. If given, `train` method must be overriden to accept these arguments.
        - Returns: A trained `torch.nn.Module`
        '''
        # ensure compiled
        assert self.__compiled is True, "[Training Error]: Manager has not yet been compiled. Either loss_fn or optimizer, or both, are not given."

        # initialize# initialize device
        cpu = torch.device("cpu")
        gpu = torch.device("cuda")
        device = gpu if torch.cuda.is_available() else cpu
        raw_model = self.model
        
        # multi gpus support
        use_multi_gpus = torch.cuda.is_available() if use_multi_gpus is True else use_multi_gpus
        if use_multi_gpus is True: self.model = torch.nn.parallel.DataParallel(self.model)
        self.model.to(device)

        # on train start
        for c in callbacks_list:
            c.on_train_start()

        # epoch loop
        for epoch in range(epochs):
            # initialize epoch
            print("Training epoch {}/{}".format(epoch+1, epochs))

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
                print(f"Epoch {epoch + 1}/{epochs}: loss={summary['loss']:.4f}, acc={summary['accuracy']:.4f}, val_loss={val_summary['loss']:.4f}, val_acc={val_summary['accuracy']:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs}: loss={summary['loss']:.4f}, acc={summary['accuracy']:.4f}")
                val_summary = None

            # on epoch end
            for c in callbacks_list:
                c.on_epoch_end(epoch, summary=summary, val_summary=val_summary)

        # load best model
        self.model = raw_model.to(cpu)
        return self.model

    @classmethod
    def from_checkpoint(cls: Type[Manager], *args, **kwargs) -> Manager:
        '''
        Method to load a manager from a saved `Checkpoint`. The manager will not be compiled with a loss function and its metrics.

        - Parameters:
            ckpt_path: A `str` of checkpoint path
        - Returns: A loaded `Manager`
        '''
        # load checkpoint
        ckpt = Checkpoint.from_saved(*args, **kwargs)
        return cls(ckpt.model, ckpt.optimizer)

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
        for _, m in self.metrics.items(): m.reset()
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
            x_train: torch.Tensor
            if use_multi_gpus is not True:
                x_train = x_train.to(device)

            # move y_train to gpu
            y_train: torch.Tensor
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
        summary = {name: float(fn.result.detach()) for name, fn in self.metrics.items()}
        return summary

    def train_step(self, x_train: torch.Tensor, y_train: torch.Tensor) -> Dict[str, float]:
        '''
        A single training step

        - Parameters:
            - x_train: The training data in `torch.Tensor`
            - y_train: The training label in `torch.Tensor`
        - Returns: A summary of `dict` with keys as `str` and values as `float`
        '''
        # forward pass
        y = self.model(x_train)
        loss = self.compiled_losses(y, y_train)
        for _, fn in self.metrics.items(): fn(y, y_train)

        # backward pass
        self.compiled_optimizer.zero_grad()
        loss.backward()
        self.compiled_optimizer.step()

        # summary result
        summary = {name: float(fn.result.detach()) for name, fn in self.metrics.items()}
        summary["loss"] = float(self.compiled_losses.result.detach())
        return summary

    def test(self, dataset: data.DataLoader, use_multi_gpus: bool=False, show_verbose: bool=False) -> Dict[str, float]:
        '''
        Test target model

        - Parameters:
            - dataset: A `data.DataLoader` to load the dataset
            - use_multi_gpus: A `bool` flag to use multi gpus during testing
        - Returns: A `dict` of validation summary
        '''
        # initialize function
        for _, m in self.metrics.items(): m.reset()
        cpu = torch.device("cpu")
        gpu = torch.device("cuda")
        device = gpu if torch.cuda.is_available() else cpu
        use_multi_gpus = torch.cuda.is_available() if use_multi_gpus is True else use_multi_gpus

        # set module status
        self.model.eval()

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
                x_test: torch.Tensor
                if use_multi_gpus is not True:
                    x_test = x_test.to(device)

                # move y_test to gpu
                y_test: torch.Tensor
                y_test = y_test.to(device)

                # test for one step
                self.test_step(x_test, y_test)

                # implement progress bar
                if progress_bar is not None:
                    progress_bar.update()

            # end epoch training
            if progress_bar is not None:
                progress_bar.close()
            
            # summarize
            summary = {name: float(fn.result.detach()) for name, fn in self.metrics.items()}
            if self.loss_fn is not None:
                summary["loss"] = float(self.compiled_losses.result.detach())
            return summary

    def test_step(self, x_test: torch.Tensor, y_test: torch.Tensor) -> None:
        '''
        A single testing step

        - Parameters:
            - x_train: The testing data in `torch.Tensor`
            - y_train: The testing label in `torch.Tensor`
        '''
        # forward pass
        y = self.model(x_test)
        for _, fn in self.metrics.items(): fn(y, y_test)

        if self.loss_fn is not None:
            self.compiled_losses(y, y_test)