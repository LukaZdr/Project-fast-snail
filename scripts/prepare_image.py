# -*- coding: utf-8 -*-

# TODO 1 take background out of given image and replace it with random background
# TODO 2 for CNN all images need to have the same size

import numpy as np
from skimage.io import filters

def binarize_image(img):
  otsu_threshold = filters.threshold_otsu(img)
  mask = img < otsu_threshold
  mask.astype(np.int)
