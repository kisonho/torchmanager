from torchmanager_core import abc, argparse, errors, os, shutil, view, yaml, _raise
from torchmanager_core.typing import Any

from .basic import Configs


class _YAML:
    yaml: str


class YAMLConfigs(Configs, abc.ABC):
    """
    A `Configs` class that can be loaded from a YAML file

    * extend: `Configs`
    * Abstract class
    """
    @classmethod
    def from_arguments(cls, *arguments: str, parser: argparse.ArgumentParser = argparse.ArgumentParser(), show_summary: bool = True):
        # get arguments
        fetched_parser = cls.get_arguments(parser=parser)
        if len(arguments) > 0:
            configs = fetched_parser.parse_args(arguments, namespace=_YAML())
        else:
            configs = fetched_parser.parse_args(namespace=_YAML())
        return cls.from_yaml(configs.yaml, show_summary=show_summary)

    @classmethod
    def from_yaml(cls, yaml_path: str, /, *, show_summary: bool = True):
        """
        Load a `Configs` directly from a YAML file

        - Parameters:
            - yaml_path: A `str` of YAML file path
            - show_summary: A `bool` of whether to show the summary of the configs
        - Returns: A formatted configuration `YAMLConfigs` object
        """
        yaml_path = os.path.normpath(yaml_path)
        assert os.path.exists(yaml_path), _raise(FileNotFoundError(f"YAML file '{yaml_path}' does not exist."))
        assert yaml_path.endswith(".yaml"), _raise(FileNotFoundError(f"YAML file '{yaml_path}' is not a valid YAML file."))

        # read yaml file
        with open(yaml_path, "r") as yaml_file:
            yaml_dict: dict[str, Any] = yaml.load(yaml_file, Loader=yaml.UnsafeLoader)

        # create configurations
        yaml_dir = os.path.dirname(yaml_path)
        yaml_dict = cls.read_extension(yaml_dict, yaml_dir)
        configs = cls(**yaml_dict)

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
    def get_arguments(parser: argparse.ArgumentParser = argparse.ArgumentParser()) -> argparse.ArgumentParser:
        """
        Get arguments from YAML file

        * classmethod

        - Parameters:
            - parser: An `ArgumentParser` object
        - Returns: An `ArgumentParser` object
        """
        parser.add_argument("yaml", type=str, help="Path to the YAML configuration file.")
        return parser

    @staticmethod
    def read_extension(yaml_dict: dict[str, Any], /, related_dir: str) -> dict[str, Any]:
        """
        Read the extension of the YAML file

        * abstractmethod

        - Parameters:
            - yaml_dict: A `dict` of the YAML file with keys as names in `str` and values as `Any` kind of values
            - related_dir: A `str` of the related directory path where the YAML file is located
        - Returns: A `dict` of the YAML file with keys as names in `str` and values as `Any` kind of values
        """
        if "extends" in yaml_dict:
            # get extensions
            extensions: list[str] = list(reversed(yaml_dict["extends"]))
            root_dict: dict[str, Any] = {}

            # read extensions
            for extension in extensions:
                extension_path = os.path.normpath(extension) if os.path.isabs(extension) else os.path.normpath(os.path.join(related_dir, extension))
                assert os.path.exists(extension_path), _raise(FileNotFoundError(f"YAML file '{extension_path}' does not exist."))
                assert extension_path.endswith(".yaml"), _raise(FileNotFoundError(f"YAML file '{extension_path}' is not a valid YAML file."))

                # read extended yaml file
                with open(extension_path, "r") as yaml_file:
                    extension_dict: dict[str, Any] = yaml.load(yaml_file, yaml.UnsafeLoader)

                # read extension for extended yaml files
                extension_dir = os.path.dirname(extension_path)
                yaml_dict = YAMLConfigs.read_extension(yaml_dict, extension_dir)

                # update yaml_dict
                root_dict.update(extension_dict)

            # update yaml_dict
            root_dict.update(yaml_dict)
            yaml_dict = root_dict
        return yaml_dict
