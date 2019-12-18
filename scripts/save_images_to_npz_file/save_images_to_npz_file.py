#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import skimage.io as io

# Import all Images
eth = io.ImageCollection("./ethernet/*")       # Label 1
vga = io.ImageCollection("./vga/*")            # Label 2
ps2 = io.ImageCollection("./ps2/*")            # Label 3
co3 = io.ImageCollection("./3comp/*")          # Label 4

# Put images in array with label
arrayOfImages = []
labels = []

for img in eth:
    arrayOfImages.append(img)
    labels.append(1)
for img in vga:
    arrayOfImages.append(img)
    labels.append(2)
for img in ps2:
    arrayOfImages.append(img)
    labels.append(3)
for img in co3:
    arrayOfImages.append(img)
    labels.append(4)
   
np.savez("vga_val", data = arrayOfImages, labels = labels)
