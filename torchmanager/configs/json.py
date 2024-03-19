from torchmanager_core import abc, argparse, errors, json, os, shutil, view, _raise
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
    @classmethod
    def from_arguments(cls, *arguments: str, parser: argparse.ArgumentParser = argparse.ArgumentParser(), show_summary: bool = True):
        # get arguments
        fetched_parser = cls.get_arguments(parser=parser)
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
        json_dir = os.path.dirname(json_path)
        json_dict = cls.read_extension(json_dict, json_dir)
        configs = cls(**json_dict)

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
        Get arguments from JSON file

        * classmethod

        - Parameters:
            - parser: An `ArgumentParser` object
        - Returns: An `ArgumentParser` object
        """
        parser.add_argument("json", type=str, help="Path to the JSON configuration file.")
        return parser

    @staticmethod
    def read_extension(json_dict: dict[str, Any], related_dir: str) -> dict[str, Any]:
        """
        Read the extension of the JSON file

        * abstractmethod

        - Parameters:
            - json_dict: A `dict` of the JSON file with keys as names in `str` and values as `Any` kind of values
            - related_dir: A `str` of the related directory path where the JSON file is located
        - Returns: A `dict` of the JSON file with keys as names in `str` and values as `Any` kind of values
        """
        if "extends" in json_dict:
            # get extensions
            extensions: list[str] = list(reversed(json_dict["extends"]))
            root_dict: dict[str, Any] = {}

            # read extensions
            for extension in extensions:
                extension_path = os.path.normpath(extension) if os.path.isabs(extension) else os.path.normpath(os.path.join(related_dir, extension))
                assert os.path.exists(extension_path), _raise(FileNotFoundError(f"JSON file '{extension_path}' does not exist."))
                assert extension_path.endswith(".json"), _raise(FileNotFoundError(f"JSON file '{extension_path}' is not a valid JSON file."))

                # read extended json files
                with open(extension_path, "r") as json_file:
                    extension_dict: dict[str, Any] = json.load(json_file)

                # read extension for extended yaml files
                extension_dir = os.path.dirname(extension_path)
                extension_dict = JSONConfigs.read_extension(extension_dict, extension_dir)

                # update json_dict
                root_dict.update(extension_dict)

            # update json_dict
            root_dict.update(json_dict)
            json_dict = root_dict
        return json_dict
