from torchmanager_core import argparse, abc, errors, os, shutil, torch, view, _raise
from torchmanager_core.typing import Optional, Union, overload
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

    def format_arguments(self) -> None:
        """Format and check current properties"""
        self.experiment = self.experiment if self.experiment.endswith(".exp") else f"{self.experiment}.exp"

    @classmethod
    def from_arguments(cls, *arguments: str, parser: argparse.ArgumentParser = argparse.ArgumentParser(), show_summary: bool = True):
        """
        Get properties from argument parser or given arguments

        * classmethod

        - Parameters:
            - arguments: Positional parameters as a `list` of arguments in `str`
            - parser: 
        - Returns: A formatted configuration object
        """
        fetched_parser = cls.get_arguments(parser=parser)
        if len(arguments) > 0:
            configs = fetched_parser.parse_args(arguments, namespace=cls())
        else:
            configs = fetched_parser.parse_args(namespace=cls())

        # format arguments
        try:
            configs.format_arguments()
        except Exception as e:
            raise errors.ConfigsFormatError(cls) from e

        # initialize logging
        log_dir = os.path.join("experiments", configs.experiment)

        # check if experiment exists
        if os.path.exists(log_dir) and configs.replace_experiment:
            shutil.rmtree(log_dir)
        elif os.path.exists(log_dir) and not configs.replace_experiment:
            raise IOError(f"Path '{log_dir}' has already existed.")

        # set log path
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.basename(configs.experiment.replace(".exp", ".log"))
        log_path = os.path.join(log_dir, log_file)
        view.set_log_path(log_path=log_path)

        # save configs
        configs.save()

        # initialize console
        formatter = view.logging.Formatter("%(message)s")
        console = view.logging.StreamHandler()
        console.setLevel(view.logging.INFO)
        console.setFormatter(formatter)
        view.logger.addHandler(console)

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

    @classmethod
    def from_experiment(cls, exp: str, /):
        """
        Load a `Configs` directly from an experiment

        - Parameters:
            - exp: A `str` of experiment path or name
        """
        exp = os.path.normpath(f"{exp}.exp") if not exp.endswith(".exp") else os.path.normpath(exp)
        cfg_path = os.path.join(exp, "configs.cfg")
        cfg = torch.load(cfg_path)
        assert isinstance(cfg, Configs), _raise(TypeError(f"Saved object at path {cfg_path} is not a valid `Configs`."))
        return cfg

    @overload
    @staticmethod
    def get_arguments(parser: argparse.ArgumentParser = argparse.ArgumentParser()) -> argparse.ArgumentParser:
        ...

    @overload
    @staticmethod
    def get_arguments(parser: argparse._ArgumentGroup) -> argparse._ArgumentGroup:
        ...

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

    def save(self) -> None:
        """Save this configuration to experiment folder"""
        exp = os.path.normpath(f"{self.experiment}.exp") if not self.experiment.endswith(".exp") else os.path.normpath(self.experiment)
        cfg_path = os.path.join("experiments", exp, "configs.cfg")
        torch.save(self, cfg_path)

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
