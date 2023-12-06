from torchmanager_core import abc, argparse, errors, os, view, yaml, _raise
from torchmanager_core.typing import Any

from .basic import Configs


class _YAML:
    yaml: str


class YAMLConfigs(Configs, abc.ABC):
    @staticmethod
    def format_yaml(yaml_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Format the YAML dictionary read from the YAML file

        * staticmethod

        - Parameters:
            - yaml_dict: A `dict` of the YAML file with keys as names in `str` and values as `Any` kind of values
        - Returns: A formatted `dict` of the YAML file with keys as names in `str` and values as `Any` kind of values
        """
        return yaml_dict

    @classmethod
    def from_arguments(cls, *arguments: str, parser: argparse.ArgumentParser = argparse.ArgumentParser(), show_summary: bool = True):
        # get arguments
        fetched_parser = cls.get_arguments(parser=parser)
        assert isinstance(fetched_parser, argparse.ArgumentParser), "Get arguments should be finished by returning an `ArgumentParser` instead of an `_ArgumentGroup`."
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
            yaml_dict: dict[str, Any] = yaml.safe_load(yaml_file)

        # create configurations
        yaml_dict = cls.format_yaml(yaml_dict)
        configs = cls(**yaml_dict)

        # format arguments
        try:
            configs.format_arguments()
        except Exception as e:
            raise errors.ConfigsFormatError(cls) from e

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
