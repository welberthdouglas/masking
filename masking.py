import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob

from sklearn.cluster import DBSCAN
from mask import get_mask

from utils import getcutout,getwcs,coord2pix
from config import IMAGES_DIR


files = glob.glob(IMAGES_DIR+"*.npz")

files_names = [file.split("/")[-1].replace(".npz","") for file in files]
data = [np.load(file)['img'] for file in files]

masks = [get_mask(img[:,:,0],pixel_threshold=250) for img in data]




