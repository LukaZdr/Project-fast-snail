# -*- coding: utf-8 -*-

# TODO 1 take background out of given image and replace it with random background
# TODO 2 for CNN all images need to have the same size

from skimage.io import imread, imsave, ImageCollection
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
import os
import random

def binarize_image(img):
  #otsu_threshold = filters.threshold_otsu(img)
  mask = img < 255
  return mask.astype(np.int)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def change_background(img, binarized_img, new_background):
  new_img = np.zeros(img.shape, dtype=int)
  for y in range(len(img)):
    for x in range(len(img[0])):
      pixel_state = binarized_image[y][x]
      if pixel_state == 0:
        bounded_y = y%len(img)
        bounded_x = x%len(img[0])
        new_img[y][x] = new_background[bounded_y][bounded_x]
      else:
        new_img[y][x] = img[y][x]
  return new_img

# loding the images from data_collection
image_collection = ImageCollection('images_to_change/*')
background_collection = ImageCollection('random_backgrounds/*')

# making a dir to save images
dir_name = 'converted_images'
dir_ending_count = 1
# finding dir name that is not taken
while os.path.exists(dir_name + '_' + str(dir_ending_count)):
  dir_ending_count += 1

current_dir = dir_name + '_' + str(dir_ending_count)
os.mkdir(current_dir) # creating dir

# transforming all images
for image_index in range(len(image_collection)):
  # binarizing the image via otsu
  image = image_collection[image_index]
  gray_image = rgb2gray(image)
  binarized_image = binarize_image(gray_image)

  # replacing the backrounf with random images
  random_background_index = random.randrange(0, len(background_collection))
  background = background_collection[random_background_index]
  img_with_new_background = change_background(image, binarized_image, background)
  plt.imshow(img_with_new_background, cmap='gray')

  # makeing a rirrored copy
  mirrored_img_with_new_background = img_with_new_background[:, ::-1]

  # saving the image
  save_path = current_dir + '/img_' + str(image_index+1) + '.jpg'
  imsave(save_path, img_with_new_background)
  save_path = current_dir + '/img_r' + str(image_index+1) + '.jpg'
  imsave(save_path, mirrored_img_with_new_background)


# plt.show()
