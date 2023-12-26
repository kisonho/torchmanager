from torchmanager_core import abc, argparse, errors, json, os, view, _raise
from torchmanager_core.typing import Any

from .basic import Configs


class _JSON:
    json: str


class JSONConfigs(Configs, abc.ABC):
    """
    A `Configs` class that can be loaded from a JSON file

    * extend: `Configs`
    * Abstract class
    """
    @staticmethod
    def format_json(json_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Format the JSON dictionary read from the JSON file

        * staticmethod

        - Parameters:
            - json_dict: A `dict` of the JSON file with keys as names in `str` and values as `Any` kind of values
        - Returns: A formatted `dict` of the JSON file with keys as names in `str` and values as `Any` kind of values
        """
        return json_dict

    @classmethod
    def from_arguments(cls, *arguments: str, parser: argparse.ArgumentParser = argparse.ArgumentParser(), show_summary: bool = True):
        # get arguments
        fetched_parser = cls.get_arguments(parser=parser)
        assert isinstance(fetched_parser, argparse.ArgumentParser), "Get arguments should be finished by returning an `ArgumentParser` instead of an `_ArgumentGroup`."
        if len(arguments) > 0:
            configs = fetched_parser.parse_args(arguments, namespace=_JSON())
        else:
            configs = fetched_parser.parse_args(namespace=_JSON())
        return cls.from_json(configs.json, show_summary=show_summary)

    @classmethod
    def from_json(cls, json_path: str, /, *, show_summary: bool = True):
        """
        Load a `Configs` directly from a JSON file

        - Parameters:
            - json_path: A `str` of JSON file path
            - show_summary: A `bool` of whether to show the summary of the configs
        - Returns: A formatted configuration `JSONConfigs` object
        """
        json_path = os.path.normpath(json_path)
        assert os.path.exists(json_path), _raise(FileNotFoundError(f"JSON file '{json_path}' does not exist."))
        assert json_path.endswith(".json"), _raise(FileNotFoundError(f"JSON file '{json_path}' is not a valid JSON file."))

        # read json file
        with open(json_path, "r") as json_file:
            json_dict: dict[str, Any] = json.load(json_file)

        # create configurations
        json_dict = cls.format_json(json_dict)
        configs = cls(**json_dict)

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
        Get arguments from JSON file

        * classmethod

        - Parameters:
            - parser: An `ArgumentParser` object
        - Returns: An `ArgumentParser` object
        """
        parser.add_argument("json", type=str, help="Path to the JSON configuration file.")
        return parser
