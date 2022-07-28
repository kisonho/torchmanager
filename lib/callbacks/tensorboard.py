from torchmanager_core import tensorboard, torch
from torchmanager_core.typing import Dict, Optional, Tuple

from .callback import Callback

class TensorBoard(Callback):
    """
    The callback to record summary to tensorboard for each epoch

    - Properties:
        - writer: A `tensorboard.SummaryWriter` to record scalars
    """
    # properties
    _writer: tensorboard.writer.SummaryWriter

    @property
    def writer(self) -> tensorboard.writer.SummaryWriter:
        return self._writer

    def __init__(self, log_dir: str) -> None:
        """
        Constructor

        - Parameters:
            - log_dir: A `str` of logging directory
        """
        super().__init__()
        self._writer = tensorboard.writer.SummaryWriter(log_dir)

    def add_graph(self, model: torch.nn.Module, input_shape: Optional[Tuple[int, ...]] = None) -> None:
        """
        Add graph to TensorBoard

        - Parameters:
            - model: A `torch.nn.Module` to add
            - input_shape: An optional `tuple` of in `int` for the inputs
        """
        inputs = torch.randn(input_shape) if input_shape is not None else None
        self._writer.add_graph(model, input_to_model=inputs)

    def on_epoch_end(self, epoch: int, summary: Dict[str, float]={}, val_summary: Optional[Dict[str, float]]=None) -> None:
        # fetch keys
        keys = list(summary.keys())
        if val_summary is not None: keys += list(val_summary.keys())
        keys = set(keys)

        # write results to Tensorboard
        for key in keys:
            result: Dict[str, float] = {}
            if key in summary:
                result["train"] = summary[key]
            if val_summary is not None and key in val_summary:
                result["val"] = val_summary[key]
            self.writer.add_scalars(key, result, epoch + 1)