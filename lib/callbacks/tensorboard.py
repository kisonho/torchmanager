from torchmanager_core import tensorboard, torch
from torchmanager_core.typing import Dict, Optional, Tuple

from .callback import FrequencyCallback
from .protocols import Frequency

class TensorBoard(FrequencyCallback):
    """
    The callback to record summary to tensorboard for each epoch

    - Properties:
        - writer: A `tensorboard.SummaryWriter` to record scalars
    """
    # properties
    __writer: tensorboard.writer.SummaryWriter

    @property
    def writer(self) -> tensorboard.writer.SummaryWriter:
        return self.__writer

    def __init__(self, log_dir: str, freq: Frequency = Frequency.EPOCH) -> None:
        """
        Constructor

        - Parameters:
            - log_dir: A `str` of logging directory
        """
        super().__init__(freq)
        self.__writer = tensorboard.writer.SummaryWriter(log_dir)

    def add_graph(self, model: torch.nn.Module, input_shape: Optional[Tuple[int, ...]] = None) -> None:
        """
        Add graph to TensorBoard

        - Parameters:
            - model: A `torch.nn.Module` to add
            - input_shape: An optional `tuple` of in `int` for the inputs
        """
        inputs = torch.randn(input_shape) if input_shape is not None else None
        self.writer.add_graph(model, input_to_model=inputs)

    def step(self, summary: dict[str, float] = {}, val_summary: Optional[dict[str, float]] = None):
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
            self.writer.add_scalars(key, result, self.current_step + 1)