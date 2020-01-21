#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

tr = np.load('./imageData_tr.npz', allow_pickle=True)
tr_images = tr['data']
tr_labels = tr['labels']

va = np.load('./imageData_va.npz', allow_pickle=True)
va_images = va['data']
va_labels = va['labels']

# Print and count all train images
# =============================================================================
# print(len(tr_images))
# for img in tr_images:
#     plt.imshow(img)
#     plt.show()
# =============================================================================

# Print and count all validating images
print(len(va_images))
for img in va_images:
    plt.imshow(img)
    plt.show()
