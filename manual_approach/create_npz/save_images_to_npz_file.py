#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import skimage.io as io

tr_or_va = "va";

# Import all Images
eth = io.ImageCollection("./ethernet_" + tr_or_va + "/*")       # Label 1
vga = io.ImageCollection("./vga_" + tr_or_va + "/*")           # Label 2
ps2 = io.ImageCollection("./ps2_" + tr_or_va + "/*")           # Label 3
co3 = io.ImageCollection("./3comp_" + tr_or_va + "/*")          # Label 4

# Put images in array with label
images = []
labels = []

if len(eth) > 0:
    for img in eth:
        images.append(img)
        labels.append(1)
if len(vga) > 0:
    for img in vga:
        images.append(img)
        labels.append(2)
if len(ps2) > 0:
    for img in ps2:
        images.append(img)
        labels.append(3)
if len(co3) > 0:
    for img in co3:
        images.append(img)
        labels.append(4)
   
np.savez("imageData_" + tr_or_va, data = images, labels = labels)
print("Done.")
