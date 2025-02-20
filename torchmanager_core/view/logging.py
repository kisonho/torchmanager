from logging import *  # type: ignore

logger = getLogger('torchmanager')

# initialize console
def add_console(console: StreamHandler = StreamHandler(), formatter = Formatter("%(message)s")) -> None:
    """
    Add console handler to the logger

    - Parameters:
        - console: A `logging.StreamHandler` of console handler
    """
    # check if console handler exists
    handlers = logger.handlers
    contains_console = any(type(handler) is type(console) for handler in handlers)

    # set console
    console.setLevel(INFO)
    console.setFormatter(formatter)

    # add console handler
    if not contains_console:
        logger.addHandler(console)
    else:
        for i, handler in enumerate(logger.handlers):
            if type(handler) is type(console):
                logger.handlers[i] = console
                break

def set_log_path(log_path: str) -> Formatter:
    """
    Set log path for torchmanager logger

    - Parameters:
        - log_path: A `str` of log file path
    - Returns: A `logging.Formatter` of log formatter with the log path
    """
    # initialize formatter
    formatter = Formatter("%(message)s")

    # loop through handlers
    for i, handler in enumerate(logger.handlers):
        if isinstance(handler, FileHandler):
            handler.baseFilename = log_path
            handler.setFormatter(formatter)
            logger.handlers[i] = handler
    return formatter

getLogger().handlers.clear()
add_console()
