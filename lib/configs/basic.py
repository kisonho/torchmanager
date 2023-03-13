from torchmanager_core import argparse, abc, os, torch, view
from torchmanager_core.typing import Any
from torchmanager_core import VERSION


class Configs(argparse.Namespace):
    """Basic Configurations"""
    experiment: str

    def __init__(self, experiment: str = NotImplemented, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.experiment = experiment if experiment.endswith(".exp") else f"{experiment}.exp"

    def _show_settings(self) -> None:
        view.logger.info("-----------Settings------------")
        view.logger.info(f"Experiments {self.experiment}")
        self.show_settings()
        view.logger.info("----------Environments----------")
        self.show_environments()
        view.logger.info("--------------------------------")

    @classmethod
    def from_arguments(cls):
        parser = cls.get_arguments()
        configs = parser.parse_args(namespace=cls())

        # initialize logging
        log_file = os.path.basename(configs.experiment.replace(".exp", ".log"))
        log_path = os.path.join(configs.experiment, log_file)
        view.set_log_path(log_path=log_path)
        return configs

    @staticmethod
    def get_arguments(parser: argparse.ArgumentParser = argparse.ArgumentParser()) -> argparse.ArgumentParser:
        parser.add_argument("-exp", "--experiments", type=str, default="test.exp")
        return parser
    
    def show_environments(self) -> None:
        view.logger.info(f"torch={torch.__version__}, torchmanager={VERSION}")
    
    @abc.abstractmethod
    def show_settings(self) -> None:
        raise NotImplementedError