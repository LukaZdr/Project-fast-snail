# -*- coding: utf-8 -*-

# TODO 1 take background out of given image and replace it with random background
# TODO 2 for CNN all images need to have the same size

from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters

def binarize_image(img):
  otsu_threshold = filters.threshold_otsu(img)
  mask = img < otsu_threshold
  return mask.astype(np.int)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

# binarizing the image via otsu
image = imread('test_img.jpg')

gray_image = rgb2gray(image)

binarized_image = binarize_image(gray_image)

plt.imshow(binarized_image, cmap='gray')

plt.show()

# replacing the backrounf with random images
background = imread('test_background.jpg')