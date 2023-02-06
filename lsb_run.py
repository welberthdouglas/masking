import numpy as np
import sys
import logging
import glob
import matplotlib.pyplot as plt

from astropy.io import fits
from tqdm import tqdm

from config import FIELDS, DATA_DIR, IMAGES_DIR, BANDS_TO_RGB, MASK_PIXEL_THRESHOLD
from download import download_fields
from img_processing import fits2rgb_processing
from mask import apply_masks
from utils import splus_conn, load_fits

file_handler = logging.FileHandler(filename="log.log", mode = "w")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
logging.basicConfig(level = logging.INFO,
                    format="%(asctime)s [%(levelname)s] - %(name)s :%(message)s",
                    datefmt="%Y-%m-%d %I:%M:%S %p",
                    handlers=[stdout_handler, file_handler])

logger = logging.getLogger(__name__)


# logger.info("Downloading data...")
# splus_connection = splus_conn()
# download_fields(fields = FIELDS, bands = ['R','G','I'], save_path = DATA_DIR, splus_connection=splus_connection)

logger.info("Loading fits files ...")
fits_files = glob.glob(DATA_DIR+"*.fits")
fields = list(set([f.replace(DATA_DIR,"").split("_")[0] for f in fits_files]))

fits_data = []
for field in tqdm(fields):
        fits_data.append([load_fits(DATA_DIR+f"{field}_band_{band}.fits") for band in BANDS_TO_RGB])


logger.info("Creating RGB images ...")
imgs = fits2rgb_processing(fits_data = fits_data,Q = 8, stretch = 3)

logger.info("Applying masks ...")
masks = []
for i in range(len(fits_data)):
    logger.info(f"Masking: {fields[i]}")
    masked = apply_masks([fits_data[i]], [imgs[i]], [fields[i]], pixel_threshold= MASK_PIXEL_THRESHOLD)
    for f,m in masked.items():
        for data,band in zip(m,BANDS_TO_RGB):
            savep = DATA_DIR + f + f"_{band}_masked.fits"
            
            fits.writeto(DATA_DIR + f + f"_{band}_masked.fits", data[0], data[1], overwrite = True) 



logger.info("Saving RGB images ...")
for img,field,mask in zip(imgs,fields,masks):
    np.savez(IMAGES_DIR + f'{field}_rgb.npz', img = img) 
    plt.imsave(IMAGES_DIR + f'{field}.png',arr=img[:,:,0] , cmap='gray_r')

# logger.info("Saving masked images ...")
# fits_data = []
# for field in tqdm(fields):
#         fits_data.append([load_fits(DATA_DIR+f"{field}_{band}_masked.fits") for band in BANDS_TO_RGB])


# imgs = fits2rgb_processing(fits_data = fits_data,Q = 8, stretch = 3)
# fits_files = glob.glob(DATA_DIR+"*_masked.fits")
# fields = set([f.replace(DATA_DIR,"").split("_")[0] for f in fits_files])
# for img,field in zip(imgs,fields):
#     np.savez(IMAGES_DIR + f'{field}_rgb.npz', img = img) 
#     plt.imsave(IMAGES_DIR + f'{field}.png',arr=img[:,:,0] , cmap='gray_r')