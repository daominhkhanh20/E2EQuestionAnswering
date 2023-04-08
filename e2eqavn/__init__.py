import logging
import sys
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)

formatted = logging.Formatter(
    "{asctime} {levelname} {name}:{lineno} - {message}", style="{"
)

stream_handler.setFormatter(formatted)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

__author__ = 'khanhdm'
__version__ = 'V1.2'
