from torchmanager_core import argparse, abc, os, shutil, torch, view, _raise
from torchmanager_core.typing import Any, Optional, Union
from torchmanager_core import DESCRIPTION


class Configs(argparse.Namespace, abc.ABC):
    """
    Basic Configurations
    
    * extends: `argparse.Namespace`
    * Abstract class

    - Properties:
        - experiment: The name in `str` of the experiment
        - replace_experiment: A `bool` flag of if replace the old experiment folder if exists

    - Method to implement:
        - show_settings: Printout current configurations, `torchmanager_core.view.logger` is recommended.
    """
    comments: Optional[str]
    experiment: str
    replace_experiment: bool

    def __init__(self, *, comments: Optional[str] = None, experiment: str = "test.exp", replace_experiment: bool = False, **kwargs: Any) -> None:
        """
        Constructor

        - Parameters:
            - experiment: A `str` of experiment name
            - replace_experiment: A `bool` flag of if replace the old experiment folder if exists
        """
        super().__init__(**kwargs)
        self.comments = comments
        self.experiment = experiment
        self.replace_experiment = replace_experiment

    def format_arguments(self) -> None:
        """Format and check current properties"""
        self.experiment = self.experiment if self.experiment.endswith(".exp") else f"{self.experiment}.exp"

    @classmethod
    def from_arguments(cls, *arguments: str, show_summary: bool = True):
        """
        Get properties from argument parser or given arguments

        * classmethod

        - Parameters:
            - arguments: Positional parameters as a `list` of arguments in `str`
        - Returns: A formatted configuration object
        """
        parser = cls.get_arguments()
        assert isinstance(parser, argparse.ArgumentParser), "Get arguments should be finished by returning an `ArgumentParser` instead of an `_ArgumentGroup`."
        if len(arguments) > 0:
            configs = parser.parse_args(arguments, namespace=cls())
        else:
            configs = parser.parse_args(namespace=cls())

        # initialize logging
        assert isinstance(configs, Configs), _raise(TypeError("The namespace is not a valid configs."))
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
        if show_summary:
            view.logger.info("------------Settings------------")
            view.logger.info(f"Experiment name: {configs.experiment}")
            configs.show_settings()
            view.logger.info("----------Environments----------")
            configs.show_environments()
            if configs.comments is not None:
                view.logger.info("------------Comments------------")
                view.logger.info(f"{configs.comments}")
            view.logger.info("--------------------------------")
        return configs

    @staticmethod
    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser()) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        """
        Set up arguments for an argument parser or argument group

        * staticmethod

        - Parameters:
            - parser: An `argparse.ArgumentParser` or `argparse._ArgumentGroup` to add arguments
        - Returns: An `argparse.ArgumentParser` or `argparse._ArgumentGroup` with arguments setup
        """
        parser.add_argument("-exp", "--experiment", type=str, default="test.exp", help="The name of experiment")
        parser.add_argument("--comments", type=str, default=None, help="The comments of this experiment.")
        parser.add_argument("--replace_experiment", action="store_true", default=False, help="The flag to replace given experiment if exists.")
        return parser

    def show_environments(self, description: str = DESCRIPTION) -> None:
        """
        Show current environments

        - Parameters:
            - description: A `str` to display the description of current app
        """
        view.logger.info(description)
        view.logger.info(f"torch={torch.__version__}")

    @abc.abstractmethod
    def show_settings(self) -> None:
        """Show current configurations"""
        pass