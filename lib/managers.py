# import typing modules
from typing import Callable, Dict, List, Optional

# import required modules
import torch
from torch.utils import data
from tqdm import tqdm

# import core modules
from .callbacks import Callback

class Manager:
    '''
    A training manager

    - Properties:
        - loss_fn: A `Callable` method that takes the truth and predictions in `torch.Tensor` and returns a loss `torch.Tensor`
        - metrics: A `dict` of metrics with a name in `str` and a `Callable` method that takes the truth and predictions in `torch.Tensor` and returns a loss `torch.Tensor`
        - model: A target `torch.nn.Module` to be trained
        - optimizer: A `torch.optim.Optimizer` to train the model
    '''
    # properties
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
    metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
    model: torch.nn.Module
    optimizer: Optional[torch.optim.Optimizer]
    
    def __init__(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer]=None, loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]=None, metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]={}) -> None:
        '''
        Constructor
        
        - Parameters:
            - loss_fn: A `Callable` method that takes the truth and predictions in `torch.Tensor` and returns a loss `torch.Tensor`
            - metrics: A `dict` of metrics with a name in `str` and a `Callable` method that takes the truth and predictions in `torch.Tensor` and returns a loss `torch.Tensor`
            - model: A target `torch.nn.Module` to be trained
            - optimizer: A `torch.optim.Optimizer` to train the model
        '''
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.model = model
        self.optimizer = optimizer

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
            - **kwargs: Additional keyword arguments that will be passed to `train_step` method. If given, `train_step` method must be overriden to accept these arguments.
        - Returns: A trained `torch.nn.Module`
        '''
        # initialize# initialize device
        cpu = torch.device("cpu")
        gpu = torch.device("cuda")
        device = gpu if torch.cuda.is_available() else cpu
        raw_model = self.model
        
        # multi gpus support
        use_multi_gpus = torch.cuda.is_available() if use_multi_gpus is True else use_multi_gpus
        if use_multi_gpus is True: self.model = torch.nn.DataParallel(self.model)
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

            # train for one batch
            summary = self.train_step(training_dataset, device=device, use_multi_gpus=use_multi_gpus, show_verbose=show_verbose, callbacks_list=callbacks_list, **kwargs)

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

    def train_step(self, dataset: data.DataLoader, device: torch.device=torch.device('cpu'), use_multi_gpus: bool=False, show_verbose: bool=False, callbacks_list: List[Callback]=[]) -> Dict[str, float]:
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
        metrics_list: Dict[str, List[torch.Tensor]] = {m: [] for m in self.metrics}
        metrics_list['loss'] = []
        self.model.train()

        # parameters assertion
        assert self.optimizer is not None, '[Training Error]: optimizer must be given during training.'
        assert self.loss_fn is not None, '[Training Error]: loss_fn function must be given during training.'

        # initialize progress bar
        if show_verbose is True:
            progress_bar = tqdm(total=len(dataset))

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

            # forward pass
            y: torch.Tensor = self.model(x_train)
            loss: torch.Tensor = self.loss_fn(y, y_train)
            metrics = {name: fn(y, y_train) for name, fn in self.metrics.items()}

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # summary result
            for m, score in metrics.items():
                metrics_list[m].append(score)
            metrics_list["loss"].append(loss)
            summary = {m: float(torch.mean(torch.Tensor(score).detach())) for m, score in metrics_list.items()}

            # on batch start
            for c in callbacks_list:
                c.on_batch_end(batch, summary=summary)

            # implement progress bar
            if show_verbose is True:
                progress_bar.set_postfix(summary)
                progress_bar.update()

        # end epoch training
        if show_verbose is True:
            progress_bar.close()

        # summarize
        summary = {m: float(torch.mean(torch.Tensor(score).detach())) for m, score in metrics_list.items()}
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
        metrics_list: Dict[str, List[torch.Tensor]] = {m: [] for m in self.metrics}
        metrics_list['loss'] = []
        cpu = torch.device("cpu")
        gpu = torch.device("cuda")
        device = gpu if torch.cuda.is_available() else cpu
        use_multi_gpus = torch.cuda.is_available() if use_multi_gpus is True else use_multi_gpus

        # set module status
        self.model.eval()

        # initialize progress bar
        if show_verbose is True:
            progress_bar = tqdm(total=len(dataset))

        # disable auto gradients
        with torch.no_grad():
            # batch loop
            for x_test, y_test in dataset:
                # move x_train to device
                x_train: torch.Tensor
                if use_multi_gpus is not True:
                    x_train = x_train.to(device)

                # move y_test to gpu
                y_test: torch.Tensor
                y_test = y_test.to(device)

                # forward pass
                y: torch.Tensor = self.model(x_test)
                metrics = {name: fn(y, y_test) for name, fn in self.metrics.items()}

                # summary result
                for m, score in metrics.items():
                    metrics_list[m].append(score)

                # add loss
                if self.loss_fn is not None:
                    loss: torch.Tensor = self.loss_fn(y, y_test)
                    metrics_list["loss"].append(loss)

                # implement progress bar
                if show_verbose is True:
                    progress_bar.update()

            # end epoch training
            if show_verbose is True:
                progress_bar.close()
            
            # summarize
            summary = {m: float(torch.mean(torch.Tensor(score).detach())) for m, score in metrics_list.items()}
            return summary
