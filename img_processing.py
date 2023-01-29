import numpy as np
from tqdm import tqdm
import glob
import logging

from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from config import DATA_DIR
from utils import load_fits

logger = logging.getLogger(__name__)

def crop_image(img:np.array, tol:float = 30)->np.array:
    """
    removes black borders from images
    """
    mask = np.where(img[:, :, 0] > tol)
    min_row, min_col = [np.min(m) for m in mask] 
    max_row, max_col = [np.max(m) for m in mask] 
    return img[ min_row:max_row, min_col:max_col]

def fits_2_rgb(fits, Q = 8, stretch = 3)-> np.array:
    """
    converts fits data to rgb images using luption transformation
    """
    g,b,r =  fits
    return make_lupton_rgb(r,g,b , Q = Q, stretch = stretch)

def fits2rgb_processing(fits_data:list, Q:float, stretch:float, flip:bool=False, crop_borders:bool =False) -> list:
    """
    performs fits preprocessing (fits2rgb + saving)
    """
    

    logger.info("Processing Fits ...")
    imgs = []
    for data in tqdm(fits_data):
        rgb = fits_2_rgb(data, Q = Q, stretch = stretch)

        if crop_borders:
            rgb = crop_image(rgb)
        
        if flip:
            img_out = np.flipud(rgb) # flipud to make it easier to compare to legacy images
        else:
            img_out = rgb
        
        imgs.append(img_out)
    
    return imgs