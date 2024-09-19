from logging import *  # type: ignore

getLogger().handlers.clear()
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
    contains_console = any(type(handler) is StreamHandler for handler in handlers)

    # add console handler
    if not contains_console:
        console.setLevel(INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

def set_log_path(log_path: str) -> Formatter:
    """
    Set log path for torchmanager logger

    - Parameters:
        - log_path: A `str` of log file path
    - Returns: A `logging.Formatter` of log formatter with the log path
    """
    logger.handlers.clear()
    logger.setLevel(INFO)
    file_handler = FileHandler(log_path)
    formatter = Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return formatter
