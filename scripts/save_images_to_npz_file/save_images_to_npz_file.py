#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import skimage.io as io
import random

# Import all Images
eth = list(io.ImageCollection("./ethernet/*"))       # Label 1 #shape 4
vga = list(io.ImageCollection("./vga/*"))            # Label 2 #shape 3
ps2 = list(io.ImageCollection("./ps2/*"))            # Label 3 #shape mixed
co3 = list(io.ImageCollection("./3comp/*"))          # Label 4 #shape 4

# Put images in array with label
arrayOfImages = []
labels = []

for img in co3:
  print(img.shape)

while (len(eth) + len(vga) + len(ps2) + len(co3)) > 0:
  next_array = random.randrange(1, 5)
  if next_array == 1 and len(eth) != 0:
    arrayOfImages.append(eth.pop(0))
    labels.append(1)
  elif next_array == 2 and len(vga) != 0:
    arrayOfImages.append(vga.pop(0))
    labels.append(2)
  elif next_array == 3 and len(ps2) != 0:
    arrayOfImages.append(ps2.pop(0))
    labels.append(3)
  elif next_array == 4 and len(co3) != 0:
    arrayOfImages.append(co3.pop(0))
    labels.append(4)
   
<<<<<<< HEAD
np.savez("vga_val", data = arrayOfImages, labels = labels)
=======
np.savez("images_and_labels", data = arrayOfImages, labels = labels)
>>>>>>> b4c4b9ca8868feb6681a66ec1fcc3535cc66f128
