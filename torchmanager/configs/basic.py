from torchmanager_core import argparse, abc, os, shutil, torch, view
from torchmanager_core.typing import Any, Union
from torchmanager_core import VERSION, DESCRIPTION


class Configs(argparse.Namespace, abc.ABC):
    """
    Basic Configurations
    
    * extends: `argparse.Namespace`
    * Abstract class

    - Properties:
        - experiment: The name in `str` of the experiment

    - Method to implement:
        - show_settings: Printout current configurations, `torchmanager_core.view.logger` is recommended.
    """
    experiment: str
    replace_experiment: bool

    def __init__(self, experiment: str = "test.exp", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.experiment = experiment

    def format_arguments(self) -> None:
        self.experiment = self.experiment if self.experiment.endswith(".exp") else f"{self.experiment}.exp"

    @classmethod
    def from_arguments(cls, *arguments: str):
        parser = cls.get_arguments()
        assert isinstance(parser, argparse.ArgumentParser), "Get arguments should be finished by returning an `ArgumentParser` instead of an `_ArgumentGroup`."
        if len(arguments) > 0:
            configs = parser.parse_args(arguments, namespace=cls())
        else:
            configs = parser.parse_args(namespace=cls())

        # initialize logging
        log_dir = os.path.join("experiments", configs.experiment)
        if os.path.exists(log_dir) and configs.replace_experiment:
            shutil.rmtree(log_dir)
        elif os.path.exists(log_dir) and not configs.replace_experiment:
            raise IOError(f"Path '{log_dir}' has already existed.")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.basename(configs.experiment.replace(".exp", ".log"))
        log_path = os.path.join(log_dir, log_file)
        view.set_log_path(log_path=log_path)
        configs.format_arguments()

        # show configs summarize
        view.logger.info("-----------Settings------------")
        view.logger.info(f"Experiment name: {configs.experiment}")
        configs.show_settings()
        view.logger.info("----------Environments----------")
        configs.show_environments()
        view.logger.info("--------------------------------")
        return configs

    @staticmethod
    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser()) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        parser.add_argument("-exp", "--experiment", type=str, default="test.exp", help="The name of experiment")
        parser.add_argument("--replace_experiment", action="store_true", default=False, help="The flag to replace given experiment if exists.")
        return parser

    def show_environments(self, description: str = DESCRIPTION) -> None:
        view.logger.info(description)
        view.logger.info(f"torch={torch.__version__}, torchmanager={VERSION}")

    @abc.abstractmethod
    def show_settings(self) -> None:
        pass