from typing import Any
from torchmanager_core import argparse, view, _raise
from torchmanager_core.typing import Any, Optional, Union

from .basic import Configs


AVOID_ARGUMENTES = ["arguments", "comments", 'experiment', "format_arguments", "from_arguments", "get_arguments", "replace_experiment", "show_environments", "show_settings"]


class DictionaryConfigs(Configs):
    """
    A configurations that saves arguments as a dictionary

    - Properties:
        - arguments: A `dict` of the arguments to be added to parser with keys as names in `str` and values as `Any` kind of default values
    """
    arguments: dict[str, Any]

    def __init__(self, *, comments: Optional[str] = None, experiment: str = "test.exp", replace_experiment: bool = False, **kwargs: Any) -> None:
        super().__init__(comments=comments, experiment=experiment, replace_experiment=replace_experiment)
        self.arguments = kwargs

    def __getattr__(self, name: str) -> Any:
        if name in self.arguments:
            return self.arguments[name]
        else:
            return super().__getattr__(name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in self.arguments:
            self.arguments[__name] = __value
        else:
            return super().__setattr__(__name, __value)

    @classmethod
    def from_arguments(cls, *arguments: str, dictionary_arguments: dict[str, Any] = {}, parser: argparse.ArgumentParser = argparse.ArgumentParser(), show_summary: bool = True):
        fetched_parser = cls.get_arguments(parser=parser, dictionary_arguments=dictionary_arguments)
        assert isinstance(fetched_parser, argparse.ArgumentParser), _raise(TypeError("The argument parser must be a top level `argparse.ArgumentParser`."))
        return super().from_arguments(*arguments, parser=fetched_parser, show_summary=show_summary)

    @staticmethod
    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser(), *, dictionary_arguments: dict[str, Any] = {}) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        """
        Set up arguments for an argument parser or argument group

        * staticmethod

        - Parameters:
            - parser: An `argparse.ArgumentParser` or `argparse._ArgumentGroup` to add arguments
            - dictionary_arguments: A `dict` of the arguments to be added to parser with keys as names in `str` and values as `Any` kind of default values
        - Returns: An `argparse.ArgumentParser` or `argparse._ArgumentGroup` with arguments setup
        """
        # initialize occupied list
        occupied_args: list[str] = AVOID_ARGUMENTES

        # loop for dictionary arguments
        for arg_name, v in dictionary_arguments.items():
            # add argument to occupied list
            if arg_name in occupied_args:
                raise ValueError(f"The argument named {arg_name} has been occupied and cannot be used, please choose another one.")
            elif arg_name.startswith("-"):
                raise ValueError(f"The argument named {arg_name} should not start with '-' or '--', please choose another one.")
            else:
                occupied_args.append(arg_name)

            # fetch help info
            if isinstance(v, tuple):
                v, h = v
                h = str(h)
            else:
                h = None

            # add to parser
            if isinstance(v, dict):
                new_arg_group = parser.add_argument_group(arg_name if h is None else f"{arg_name}: {h}")
                new_arg_group = DictionaryConfigs.get_arguments(new_arg_group, dictionary_arguments=v)
            elif isinstance(v, float):
                parser.add_argument(f"--{arg_name}", type=float, default=v, help=h)
            elif isinstance(v, int):
                parser.add_argument(f"--{arg_name}", type=int, default=v, help=h)
            elif isinstance(v, list):
                parser.add_argument(f"--{arg_name}", nargs="+", default=v, help=h)
            elif isinstance(v, str):
                parser.add_argument(f"--{arg_name}", type=str, default=v, help=h)
            else:
                parser.add_argument(f"--{arg_name}", default=v, help=h)

        # fetch basic arguments
        return Configs.get_arguments(parser)

    def show_settings(self) -> None:
        # initialize info
        info = ""

        # loop for dictionary arguments
        for i, (arg_name, v) in enumerate(self.arguments.items()):
            if i > 0:
                info += ", "
            info += f"{arg_name}={v}"

        # log settings
        view.logger.info(info)
        super().show_settings()
