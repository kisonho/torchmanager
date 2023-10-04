import multiprocessing, traceback
from typing import Any, Callable, Optional

class Process(multiprocessing.Process):
    """
    A process that can return exception

    * extends: `multiprocessing.Process`

    - Properties:
        - exception: An optional `Exception` of the exception raised in the process
    """
    __exception: Optional[Exception]

    def __init__(self, func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> None:
        """
        Constructor

        - Parameters:
            - func: The function to run in the process
            - args: The positional arguments to pass to the function
            - kwargs: The keyword arguments to pass to the function
        """
        super().__init__(target=func, args=args, kwargs=kwargs)
        self.__exception = None

    def run(self) -> None:
        try:
            super().run()
        except Exception as e:
            self.__exception = e
            pass

    @property
    def exception(self) -> Optional[Exception]:
        e = self.__exception
        self.__exception = None
        return e
