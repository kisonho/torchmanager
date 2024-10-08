from torchmanager.callbacks import FrequencyCallback
from torchmanager_core import tensorboard, torch
from torchmanager_core.typing import Optional, Set
from torchmanager_core.protocols import Frequency



class TensorBoard(FrequencyCallback):
    """
    The callback to record summary to tensorboard for each epoch

    * extends: `.callback.FrequencyCallback`
    * requires: `tensorboard` package

    - Properties:
        - writer: A `tensorboard.SummaryWriter` to record scalars
    """
    # properties
    __writer: tensorboard.writer.SummaryWriter

    @property
    def writer(self) -> tensorboard.writer.SummaryWriter:
        return self.__writer

    def __init__(self, log_dir: str, freq: Frequency = Frequency.EPOCH, initial_step: int = 0) -> None:
        """
        Constructor

        - Parameters:
            - log_dir: A `str` of logging directory
            - freq: A tensorboard record `.protocols.Frequency`
            - initial_step: An `int` of the initial step that starts with
        """
        super().__init__(freq, initial_step=initial_step)
        self.__writer = tensorboard.writer.SummaryWriter(log_dir)
        assert self.freq == Frequency.EPOCH or self.freq == Frequency.BATCH, "Record to tensorboard at start of batch or epoch is not supported."

    def add_graph(self, model: torch.nn.Module, input_shape: Optional[tuple[int, ...]] = None) -> None:
        """
        Add graph to TensorBoard

        - Parameters:
            - model: A `torch.nn.Module` to add
            - input_shape: An optional `tuple` of in `int` for the inputs
        """
        inputs = torch.randn(input_shape) if input_shape is not None else None
        self.writer.add_graph(model, input_to_model=inputs)

    def _update(self, result: tuple[Set[str], dict[str, float], Optional[dict[str, float]]]) -> None:
        keys, summary, val_summary = result

        # write results to Tensorboard
        for key in keys:
            # initialize
            r: dict[str, float] = {}

            # record key
            if key in summary:
                r["train"] = summary[key]
            if val_summary is not None and key in val_summary:
                r["val"] = val_summary[key]
            self.writer.add_scalars(key, r, self.current_step + 1)

    def step(self, summary: dict[str, float], val_summary: Optional[dict[str, float]] = None) -> tuple[Set[str], dict[str, float], Optional[dict[str, float]]]:
        # fetch keys
        keys = list(summary.keys())
        if val_summary is not None:
            keys += list(val_summary.keys())
        keys = set(keys)
        return keys, summary, val_summary
