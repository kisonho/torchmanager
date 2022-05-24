from __future__ import annotations
from torchmanager_core import devices, torch
from torchmanager_core.typing import Any, Dict, Generic, Optional, OrderedDict, Type, TypeVar

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
    # properties
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
            - model: A target `torch.nn.Module`
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
    def from_saved(cls: Type[Checkpoint[Any]], ckpt_path: str, model: Optional[torch.nn.Module] = None) -> Checkpoint[Any]:
        '''
        Load checkpoint from a saved checkpoint file

        - Parameters:
            - ckpt_path: A `str` of file path
            - model: An optional `torch.nn.Module` to be loaded
        '''
        # load checkpint dictionary
        ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location=devices.CPU)

        # load model
        if ckpt["save_weights_only"] is True:
            assert model is not None, "[Checkpoint Error]: The structure model is not given."
            state_dict: OrderedDict[str, torch.Tensor] = ckpt["model"]
            model.load_state_dict(state_dict)
            ckpt["model"] = model
        else:
            # remove data parallel wrap
            if isinstance(ckpt["model"], torch.nn.parallel.DataParallel):
                ckpt["model"] = ckpt["model"].module

            # load model structure with checkpoint weights
            if model is not None:
                assert isinstance(ckpt["model"], torch.nn.Module), "[Ckpt Error]: This saved model is not a valid PyTorch module. It cannot be loaded as weights into a given model."
                model.load_state_dict(ckpt["model"].state_dict())
                ckpt["model"] = model
        return cls(**ckpt)

    def save(self, epoch: int, ckpt_path: str) -> None:
        self.last_epoch = epoch
        ckpt = self.__dict__
        if self.save_weights_only is True:
            assert isinstance(self.model, torch.nn.Module), "[Ckpt Error]: This saved model is not a valid PyTorch module. It cannot be loaded with weights only."
            ckpt["model"] = self.model.state_dict()
        torch.save(ckpt, ckpt_path)