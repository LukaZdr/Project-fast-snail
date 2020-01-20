# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 22:28:46 2020

@author: MightyA
"""


import matplotlib.pyplot as plt
from skimage.io import imread, imshow, ImageCollection, imsave
from skimage.transform import resize
import skimage.io as io
import glob

allow_pickle = True

from tempfile import TemporaryFile
outfile = TemporaryFile()

image_collection = ImageCollection('bilder/*.png')

croped_images = []

for img in image_collection:
    croped_images.append(resize(img, (255,255)))
    
img_count = 1
for cropted_img in croped_images:
    imsave('result/img' + str(img_count) + ".png", cropted_img)
    img_count += 1
    
    