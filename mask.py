import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from astropy.io import fits
from tqdm import tqdm

def get_bright_pixels(img:np.array, threshold:float = 255)->np.array:
    """
    set to zero all pixels smaller than threshold
    """
    
    return np.where(img < threshold, 0, img)

def get_bright_objects_labels(labeled_img:np.array, bright_objects_slices:list)->list:
    """
    takes the labeled image and bright object slices and return a set with all the objects id in the labeled img
    corresponding to the bright objects slices locations
    """ 
    obj_ids = [set(labeled_img[s].flatten()) for s in bright_objects_slices]
    set_union = set().union(*obj_ids)
    set_union.remove(0)
    
    return list(set_union)

def get_mask(image:np.array, pixel_threshold:float)->np.array:
    """
    returns a mask for img using pixel_threshold as threshold for detecting bright objects
    """
    labeled_img,_ = ndimage.label(image)
    structure = ndimage.generate_binary_structure(2,2)
    
    bright_objects = get_bright_pixels(image,pixel_threshold)
    labeled_bright_objects,_ = ndimage.label(bright_objects, structure)
    bright_objects_slices = ndimage.find_objects(labeled_bright_objects)
    
    mask = labeled_img.copy()
    bright_obj_ids = get_bright_objects_labels(labeled_img, bright_objects_slices)
    out = np.where(np.isin(mask,bright_obj_ids),1,0)
    
    return 1-binary_fill_holes(out)


def apply_masks(fits_file:list, imgs:list, fields:str, pixel_threshold:float) -> dict:
    masked_fields = {}
    for f,im,field in zip(fits_file, imgs,fields):
        masked_field = []
        for i in range(len(f)):
            mask = get_mask(im[:,:,i],pixel_threshold=pixel_threshold)
            data = f[0].data*mask
            header = f[0].header
            masked_field.append((data, header))
        masked_fields[field] = masked_field
    
    return masked_fields