import os
import splusdata
import logging

from astropy.io import fits
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_fits_splus(field, conn, save_path:str, bands:list) -> None:
    """
    downloads fits files of the specified bands from splus and saves in the specified path
    """
    for band in bands:
        fz_file = conn.get_field(field, band)
        data = fz_file[1].data
        header = fz_file[1].header
        fits.writeto(f"{save_path}{field}_band_{band}.fits",header = header, data=data, overwrite=True)  


def download_fields(fields:list, save_path:str, bands:list, splus_connection:splusdata.connect) -> None:
    """
    download fits files for RGI bandss
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    logger.info("Downloading fields ...")
    for field in tqdm(fields):
        get_fits_splus(field, splus_connection, save_path, bands = bands)