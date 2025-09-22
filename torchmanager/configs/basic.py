from torchmanager_core import argparse, abc, errors, os, platform, shutil, sys, torch, view, _raise, DESCRIPTION
from torchmanager_core.typing import Optional, Union, overload


class BaseConfigs(argparse.Namespace, abc.ABC):
    """
    Basic Configurations
    
    * extends: `argparse.Namespace`
    * abstract class

    - abstract methods
        - format_arguments: Format and check current properties
        - (classmethod) from_arguments: Get properties from argument parser or given arguments
        - (static) get_arguments: Set up arguments for an argument parser or argument group
        - show_settings: Printout current configurations, `torchmanager_core.view.logger` is recommended.
    """
    @abc.abstractmethod
    def format_arguments(self) -> None:
        """Format and check current properties"""
        ...

    @classmethod
    @abc.abstractmethod
    def from_arguments(cls: type["BaseConfigs"], *arguments: str, parser: argparse.ArgumentParser = argparse.ArgumentParser(), show_summary: bool = True) -> "BaseConfigs":
        """
        Get properties from argument parser or given arguments

        * classmethod

        - Parameters:
            - arguments: Positional parameters as a `list` of arguments in `str`
            - parser: 
        - Returns: A formatted configuration object
        """
        ...

    @classmethod
    def from_experiment(cls: type["BaseConfigs"], exp: str, /) -> "BaseConfigs":
        """
        Load a `Configs` directly from an experiment

        - Parameters:
            - exp: A `str` of experiment path or name
        """
        exp = os.path.normpath(f"{exp}.exp") if not exp.endswith(".exp") else os.path.normpath(exp)
        cfg_path = os.path.join(exp, "configs.cfg")
        cfg = torch.load(cfg_path)
        assert isinstance(cfg, BaseConfigs), _raise(TypeError(f"Saved object at path {cfg_path} is not a valid `Configs`."))
        return cfg

    @overload
    @staticmethod
    @abc.abstractmethod
    def get_arguments(parser: argparse.ArgumentParser = argparse.ArgumentParser()) -> argparse.ArgumentParser:
        ...

    @overload
    @staticmethod
    @abc.abstractmethod
    def get_arguments(parser: argparse._ArgumentGroup) -> argparse._ArgumentGroup:
        ...

    @staticmethod
    @abc.abstractmethod
    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser()) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        """
        Set up arguments for an argument parser or argument group

        * staticmethod

        - Parameters:
            - parser: An `argparse.ArgumentParser` or `argparse._ArgumentGroup` to add arguments
        - Returns: An `argparse.ArgumentParser` or `argparse._ArgumentGroup` with arguments setup
        """
        ...

    def show_environments(self, description: str = DESCRIPTION) -> None:
        """
        Show current environments

        - Parameters:
            - description: A `str` to display the description of current app
        """
        view.logger.info(description)
        view.logger.info(f"python={sys.version}")
        view.logger.info(f"torch={torch.__version__}")
        view.logger.info(f"platform={platform.platform()}")

    @abc.abstractmethod
    def show_settings(self) -> None:
        """Show current configurations"""
        pass


class Configs(argparse.Namespace, abc.ABC):
    """
    Basic Configurations
    
    * extends: `argparse.Namespace`
    * abstract methods: `show_settings`

    - Properties:
        - comments: The comments of this experiment
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

    @classmethod
    def from_arguments(cls: type["Configs"], *arguments: str, parser: argparse.ArgumentParser = argparse.ArgumentParser(), show_summary: bool = True) -> "Configs":
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
        formatter = view.set_log_path(log_path=log_path)
        view.add_console(formatter=formatter)

        # save configs
        configs.save()

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
        view.logger.info(f"python={sys.version}")
        view.logger.info(f"torch={torch.__version__}")
        view.logger.info(f"platform={platform.platform()}")

    @abc.abstractmethod
    def show_settings(self) -> None:
        """Show current configurations"""
        pass
