# import typing modules
from __future__ import annotations
from typing import Any, Optional, Type

# import required modules
import abc, argparse, os

# import core modules
from ...lib.train.callbacks import TensorBoard

class Config(abc.ABC):
    """An abstract configuration class"""
    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @classmethod
    def from_arguments(cls: Type[Config], parser: argparse.ArgumentParser = argparse.ArgumentParser(), **kwargs: Any) -> Config:
        """Get configurations from arugments"""
        cls.set_arguments(parser)
        arguments = parser.parse_args().__dict__
        kwargs.update(arguments)
        return cls(**kwargs)

    @abc.abstractmethod
    def show_arguments(self) -> None:
        """Print out all arguments"""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def set_arguments(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
        """
        Set available arugments

        - Parameters:
            - parser: An `argparser.ArgumentParser` to add the arguments
        - Returns: An `argparser._ArgumentGroup` of the current configuration
        """
        raise NotImplementedError


class ExperimentConfig(Config):
    """
    Configurations for an experiment
    
    - Properties:
        - experiment: An optional `str` of experiment name
        - show_verbose: A `bool` flag of if showing the progress bar
        - use_multi_gpus: A `bool` flag of if using multi gpus
    """
    experiment: Optional[str] = None
    show_verbose: bool = False
    use_multi_gpus: bool = False

    def __init__(self, show_verbose: bool = False, use_multi_gpus: bool = False, experiment: str = "None") -> None:
        self.show_verbose = show_verbose
        self.use_multi_gpus = use_multi_gpus
        if experiment.lower() != "none":
            self.experiment = f"{experiment}.exp" if not experiment.endswith(".exp") else experiment

    @staticmethod
    def set_arguments(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
        exp_args = parser.add_argument_group("Experiment arguments")
        exp_args.add_argument("--experiment", type=str, default="None", help="Name of experiment, default is \'None\'.")
        exp_args.add_argument("--use_multi_gpus", action="store_true", default=False, help="Flag to use multi gpus.")
        exp_args.add_argument("--show_verbose", action="store_true", default=False, help="Flag to show verbose.")
        return exp_args


class TrainingConfig(ExperimentConfig):
    """
    Configurations for training

    - Properties:
        - epochs: An `int` of total training epochs
        - initial_epoch: An `int` of the starting epoch index
        - lr_decay: A `float` of the learning rate decay rate
        - lr_decay_step: An `int` of the learning rate decay step
    """
    epochs: int
    initial_epoch: int
    lr_decay: float
    lr_decay_step: int

    def __init__(self, epochs: int, initial_epoch: int = 0, lr_decay: float = -1, lr_decay_step: int = 1, **kwargs) -> None:
        """Constructor"""
        super().__init__(**kwargs)
        # set arguments
        self.epochs = epochs
        self.initial_epoch = initial_epoch
        self.lr_decay = lr_decay
        self.lr_decay_step = lr_decay_step

        # assert arugments
        assert self.epochs > 0, f"[Config Error]: Epochs must be a positive number, got {self.epochs}."
        assert self.initial_epoch >= 0, f"[Config Error]: Initial epoch must be a non-negative number, got {self.initial_epoch}."
        assert self.lr_decay_step > 0, f"[Config Error]: The decay step must be a positive number, got {self.lr_decay_step}."
        self.show_arguments()

    def show_arguments(self) -> None:
        print(f"Training settings: epochs={self.epochs}, initial_epoch={self.initial_epoch}, use_multi_gpus={self.use_multi_gpus}")
        print(f"Device settings: use_multi_gpus={self.use_multi_gpus}")

    @staticmethod
    def set_arguments(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
        training_args = parser.add_argument_group("Training arguments")
        training_args.add_argument("-e", "--epochs", type=int, default=1, help="The total number of training epochs, default is 1.")
        training_args.add_argument("--initial_epoch", type=int, default=0, help="The number of initial epoch, default is 0.")
        training_args.add_argument("--lr_decay", type=float, default=-1, help="The learning rate decay, a non-positive number or not given will using constant schedule, default is -1.")
        training_args.add_argument("--lr_decay_step", type=int, default=1, help="The learning rate decay frequency in epochs, default is 1.")
        super().set_arguments(parser)
        return training_args