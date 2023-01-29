import numpy as np
from tqdm import tqdm
import glob
import logging

from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from config import *

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

def fits2rgb_processing(load_path:str, save_path:str, Q:float, stretch:float, bands:list, flip:bool=False, crop_borders:bool =False) -> None:
    """
    performs fits preprocessing (fits2rgb + saving)
    """

    fits_files = glob.glob(load_path+"*.fits")
    fields = set([f.replace(DATA_DIR,"").split("_")[0] for f in fits_files])
    

    logger.info("Processing Fits ...")
    for field in tqdm(fields):
        fits_data = [fits.open(DATA_DIR+f"{field}_band_{band}.fits", memmap=False)[0].data for band in bands]
        rgb = fits_2_rgb(fits_data, Q = Q, stretch = stretch)

        if crop_borders:
            rgb = crop_image(rgb)
        
        if flip:
            img_out = np.flipud(rgb) # flipud to make it easier to compare to legacy images
        else:
            img_out = rgb
        np.savez(save_path + f'{field}_rgb.npz', img = img_out) 