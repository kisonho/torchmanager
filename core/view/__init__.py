import logging, warnings
from tqdm import tqdm

from .protocols import VerboseType

logger = logging.getLogger('torchmanager')

def set_log_path(log_path: str) -> logging.Formatter:
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return formatter