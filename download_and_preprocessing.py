import logging
import sys

from img_processing import fits2rgb_processing
from utils import download_fields
from config import FIELDS, DATA_DIR, IMAGES_DIR

file_handler = logging.FileHandler(filename="log.log", mode = "w")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
logging.basicConfig(level = logging.INFO,
                    format="%(asctime)s [%(levelname)s] - %(name)s :%(message)s",
                    datefmt="%Y-%m-%d %I:%M:%S %p",
                    handlers=[stdout_handler, file_handler])

logger = logging.getLogger("__name__")



download_fields(fields = FIELDS, bands = ['R','G','I'], save_path = DATA_DIR)
fits2rgb_processing(load_path = DATA_DIR, save_path = IMAGES_DIR,Q = 8, stretch = 3, bands = ['R','G','I'])