from logging import *  # type: ignore

getLogger().handlers.clear()
logger = getLogger('torchmanager')

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
