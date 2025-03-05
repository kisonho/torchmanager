from logging import *  # type: ignore

logger = getLogger('torchmanager')
logger.setLevel(INFO)

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
    handlers = logger.handlers
    formatter = Formatter("%(message)s")
    contains_file_handler = any(isinstance(handler, FileHandler) for handler in handlers)

    # loop through handlers
    if contains_file_handler:
        for i, handler in enumerate(handlers):
            if isinstance(handler, FileHandler):
                handler.baseFilename = log_path
                handler.setFormatter(formatter)
                logger.handlers[i] = handler
                break
    else:
        file_handler = FileHandler(log_path)
        file_handler.setLevel(INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return formatter
