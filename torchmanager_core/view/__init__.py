import logging, warnings
from tqdm import tqdm

from .protocols import VerboseType

logging.getLogger().handlers.clear()
logger = logging.getLogger('torchmanager')