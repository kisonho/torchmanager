from torchmanager_core import devices, torch, _raise
from torchmanager_core.typing import Any, Dict, Generic, Optional, OrderedDict, TypeVar

from .protocols import StateDictLoadable

T = TypeVar('T')

class Checkpoint(Generic[T]):
    '''
    The callback to save the latest checkpoint for each epoch

    - Properties:
        - last_epoch: An `int` of the last epoch index
        - loss_fn: An optional `Loss` object
        - metrics: An optional `dict` with name in `str` and value of the `Metric` objects for metrics
        - model: Any type of model to be saved
        - optimizer: A `torch.nn.Optimizer` to be saved
        - save_weights_only: A `bool` flag of if only save state_dict of model
    '''
    last_epoch: int
    loss_fn: Optional[torch.nn.Module]
    metrics: Dict[str, torch.nn.Module]
    model: T
    optimizer: Optional[torch.optim.Optimizer]
    save_weights_only: bool

    def __init__(self, model: T, last_epoch: int=0, optimizer: Optional[torch.optim.Optimizer]=None, loss_fn: Optional[torch.nn.Module]=None, metrics: Optional[Dict[str, torch.nn.Module]]=None, save_weights_only: bool=False) -> None:
        '''
        Constructor

        - Parameters:
            - model: Any type of model to be saved
            - epoch: An `int` of epoch index
            - optimizer: An optional `torch.optim.Optimizer` to be recorded
            - loss_fn: An optional `torch.nn.Module` for loss function to be recorded
            - metrics: An optional `dict` of the metrics with key in `str` and value in `torch.nn.Module` to be recorded
        '''
        super().__init__()
        self.last_epoch = last_epoch
        self.loss_fn = loss_fn
        self.metrics = metrics if metrics is not None else {}
        self.model = model
        self.optimizer = optimizer
        self.save_weights_only = save_weights_only

    @classmethod
    def from_saved(cls, ckpt_path: str, model: Any = None):
        '''
        Load checkpoint from a saved checkpoint file

        - Parameters:
            - ckpt_path: A `str` of file path
            - model: Any object to be loaded
        '''
        # load checkpint dictionary
        ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location=devices.CPU)
        assert isinstance(model, StateDictLoadable) or model is None, _raise(TypeError("[Checkpoint Error]: The given model does not perform to `StateDictLoadable` protocol."))

        # load model
        if ckpt["save_weights_only"] is True:
            state_dict: OrderedDict[str, Any] = ckpt["model"]
            model.load_state_dict(state_dict)
            ckpt["model"] = model
        else:
            # remove data parallel wrap
            if isinstance(ckpt["model"], torch.nn.parallel.DataParallel):
                ckpt["model"] = ckpt["model"].module

            # load model structure with checkpoint weights
            if model is not None:
                assert isinstance(ckpt["model"], StateDictLoadable), _raise(TypeError("The saved model does not perform to `StateDictLoadable` protocol. It cannot be loaded into the given model"))
                model.load_state_dict(ckpt["model"].state_dict())
                ckpt["model"] = model
        return cls(**ckpt)

    def save(self, epoch: int, ckpt_path: str) -> None:
        """
        Saves current checkpoint

        - Parameters:
            - epoch: The `int` index of current epoch to save
            - ckpt_path: The `str` of checkpoint path to save
        """
        self.last_epoch = epoch
        ckpt = self.__dict__
        if self.save_weights_only is True:
            assert isinstance(self.model, StateDictLoadable), _raise(TypeError("This saved model does not perform to `StateDictLoadable` protocol to save weights only."))
            ckpt["model"] = self.model.state_dict()
        elif isinstance(ckpt["model"], torch.nn.parallel.DataParallel):
            ckpt["model"] = ckpt["model"].module
        torch.save(ckpt, ckpt_path)