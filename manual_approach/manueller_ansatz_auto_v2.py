# -*- coding: utf-8 -*-
import numpy as np
from skimage.feature import hog
from skimage.filters import sobel_h, sobel_v
from skimage.color import rgb2gray
from sklearn.utils import shuffle
from timeit import default_timer
from csv import writer, QUOTE_MINIMAL
#import matplotlib.pyplot as plt

# Define needed functions
# Histrogram 1D (gray)
def rgb_img_to_1d_histo(img, bin_count):
	img = img.ravel() # ravel => returns flattend 1d-Array
	return np.histogram(img, bins=bin_count, range=(0, 256))

# Histrogram 3D (color)
def rgb_img_to_3d_histo(img, bin_count):
	img = img.reshape(img.shape[0] * img.shape[1], 3)
	return np.histogramdd(img, bins=[bin_count, bin_count, bin_count], range=((0, 256), (0, 256), (0, 256)))

# Mean value of image
def rgb_img_mean(img): 
	return np.mean(img, axis = (0, 1, 2))

# Returns indices of n smallest elements of array
def indices_of_n_smallest_values(neighbors, num_list):
	return np.array(num_list).argsort()[:neighbors]

# Returns indices of n largest elements of array
def indices_of_n_biggest_values(n, num_list):
	return np.array(num_list).argsort()[-n:][::-1]

# Returns elements of indices of a list of indices and a list of elements
def elements_for_index_list(index_list, elements_list):
	return list(map(lambda x: elements_list[x], index_list))

# Returns the most frequently occurring element
def most_common_occurrence(element_list):
	return max(element_list, key=element_list.count)

# Euklid distance measure
def euklid(x, y):
	return np.sqrt(np.sum((x - y)**2))

# Intersection distance measure
def intersection(x, y):
    intersection = np.true_divide(np.sum(np.minimum(x, y)), np.sum(y))
    return intersection

# Takes distance of up to 2 descriptors and sums up
def calculate_combined_weighted_distance(img_1, img_2, distance_measure, neighbour_count, descriptor_1, descriptor_2, weight, bin_count):
	# Set Descriptor 1
    if descriptor_1 == "1d_histo":
        deskr_1_img_1 = rgb_img_to_1d_histo(img_1, bin_count)[0]
        deskr_1_img_2 = rgb_img_to_1d_histo(img_2, bin_count)[0]
    elif descriptor_1 == "3d_histo":
        deskr_1_img_1 = rgb_img_to_3d_histo(img_1, bin_count)[0]
        deskr_1_img_2 = rgb_img_to_3d_histo(img_2, bin_count)[0]
    elif descriptor_1 == "std": # (Standardabweichung)
        deskr_1_img_1 = np.std(img_1)
        deskr_1_img_2 = np.std(img_2)
    elif descriptor_1 == "mean": # (Mittelwert)
        deskr_1_img_1 = rgb_img_mean(img_1)
        deskr_1_img_2 = rgb_img_mean(img_2)
    elif descriptor_1 == "sobel": # Find edges
        sobelH_1 = sobel_h(rgb2gray(img_1)) # Find horizontal edges/brightness differences
        sobelV_1 = sobel_v(rgb2gray(img_1)) # Find vertical edges/brightness differences
        deskr_1_img_1 = np.sqrt(sobelH_1**2 + sobelV_1**2) # Gradient strength
        sobelH_2 = sobel_h(rgb2gray(img_2))
        sobelV_2 = sobel_v(rgb2gray(img_2))
        deskr_1_img_2 = np.sqrt(sobelH_2**2 + sobelV_2**2)
    elif descriptor_1[:3] == "hog": # Histogram over gradient orientation
        params = descriptor_1.split(",")
        par1 = int(params[1])
        par2 = int(params[2])
        deskr_1_img_1 = hog(img_1, orientations=par1, pixels_per_cell=(par2, par2))
        deskr_1_img_2 = hog(img_1, orientations=par1, pixels_per_cell=(par2, par2))
        # With image return:
        # deskr_1_img_1, image1 = hog(img_1, orientations=par1, pixels_per_cell=(par2, par2), visualize=True)
        # deskr_1_img_2, image2 = hog(img_2, orientations=par1, pixels_per_cell=(par2, par2), visualize=True)
        # plt.imshow(image2)
        # plt.show()
    elif descriptor_1 == "0":
        deskr_1_img_1 = 0
        deskr_1_img_2 = 0
        
	# Set Descriptor 2
    if descriptor_2 == "1d_histo":
        deskr_2_img_1 = rgb_img_to_1d_histo(img_1, bin_count)[0]
        deskr_2_img_2 = rgb_img_to_1d_histo(img_2, bin_count)[0]
    elif descriptor_2 == "3d_histo":
        deskr_2_img_1 = rgb_img_to_3d_histo(img_1, bin_count)[0]
        deskr_2_img_2 = rgb_img_to_3d_histo(img_2, bin_count)[0]
    elif descriptor_2 == "std":
        deskr_2_img_1 = np.std(img_1)
        deskr_2_img_2 = np.std(img_2)
    elif descriptor_2 == "mean":
        deskr_2_img_1 = rgb_img_mean(img_1)
        deskr_2_img_2 = rgb_img_mean(img_2)
    elif descriptor_2 == "sobel":
        sobelH_1 = sobel_h(rgb2gray(img_1))
        sobelV_1 = sobel_v(rgb2gray(img_1))
        deskr_2_img_1 = np.sqrt(sobelH_1**2 + sobelV_1**2)
        sobelH_2 = sobel_h(rgb2gray(img_2))
        sobelV_2 = sobel_v(rgb2gray(img_2))
        deskr_2_img_2 = np.sqrt(sobelH_2**2 + sobelV_2**2)
    elif descriptor_2[:3] == "hog": 
        params = descriptor_2.split(",")
        par1 = int(params[1])
        par2 = int(params[2])
        deskr_2_img_1 = hog(img_1, orientations=par1, pixels_per_cell=(par2, par2))
        deskr_2_img_2 = hog(img_1, orientations=par1, pixels_per_cell=(par2, par2))
    elif descriptor_2 == "0":
        deskr_2_img_1 = 0
        deskr_2_img_2 = 0
        
    # Calculate distance
    if distance_measure == "euklid":
        distance_1 = euklid(deskr_1_img_1, deskr_1_img_2)
        distance_2 = euklid(deskr_2_img_1, deskr_2_img_2)
    elif distance_measure == "intersect":
        distance_1 = intersection(deskr_1_img_1, deskr_1_img_2)
        distance_2 = intersection(deskr_2_img_1, deskr_2_img_2)
        
    return distance_1 + weight * distance_2

# Calculates esimated labels through nearest neighbour method
def rgb_img_n_nearest_neighbour(run_nr, image_set, va_imgs, tr_imgs, tr_labels, distance_measure, neighbour_count, descriptor_1, descriptor_2, weight, bin_count):
    print("\nRUN " + str(run_nr) + ": ImgSet " + str(image_set) + " / " + distance_measure + " / " + str(neighbour_count) + " / " + descriptor_1 + " / " + descriptor_2 + " / " + str(weight) + " / " + str(bin_count))
    wait_length = str(len(va_imgs))
    est_labels_list = []
    for i, va_img in enumerate(va_imgs):
        distances = []
        for tr_img in tr_imgs:                            
            distance = calculate_combined_weighted_distance(va_img, tr_img, distance_measure, neighbour_count, descriptor_1, descriptor_2, weight, bin_count)
            distances.append(distance)
        index_list = indices_of_n_smallest_values(neighbour_count, distances)
        n_closest_labels = elements_for_index_list(index_list, tr_labels)
        est_label = most_common_occurrence(n_closest_labels)
        est_labels_list.append(est_label)
        print(str(i + 1) + "/" + wait_length)
    return est_labels_list

# Guessing accuracy in %
def guessing_accuracy(est_labels, val_labels):
    count = 0
    for index in range(len(est_labels)):
        if est_labels[index] == val_labels[index]:
            count += 1
    return count / len(val_labels) * 100

# Calculate estimated labels and write to csv file (with # for comment-lines)
def run(run_nr, image_set, distance_measure, neighbour_count, descriptor_1, descriptor_2, weight, bin_count):
    # Load data
    if image_set == 1:   # Train images with own backgrounds + original validation images
        tr_images = tr1["data"]
        tr_labels = tr1["labels"]
        va_images = va1["data"]
        va_labels = va1["labels"]
    elif image_set == 2: # Train images and validation images both with own backgrounds
        tr_images = tr2["data"]
        tr_labels = tr2["labels"]
        va_images = va2["data"]
        va_labels = va2["labels"]
    elif image_set == 3: # Train images with white background + original validation images
        tr_images = tr3["data"]
        tr_labels = tr3["labels"]
        va_images = va3["data"]
        va_labels = va3["labels"]
    
    # Random shuffle images
    tr_shuffled_images, tr_shuffled_labels = shuffle(tr_images, tr_labels)
    va_shuffled_images, va_shuffled_labels = shuffle(va_images, va_labels)
    
    # Start time measurement
    time_start = default_timer()
    
    # Calculate esimated labels
    estimated_labels = rgb_img_n_nearest_neighbour(run_nr, image_set, va_shuffled_images, tr_shuffled_images, tr_shuffled_labels, distance_measure, neighbour_count, descriptor_1, descriptor_2, weight, bin_count)
    
    # Stop time measurement (in min)
    time_stop = default_timer()
    time_needed = (time_stop - time_start) / 60
    
    # Calculate guessing accuracy
    guessing_acc = guessing_accuracy(estimated_labels, va_shuffled_labels)
    
    # Write settings and result to file
    with open("Results.csv", mode="a", newline="\n", encoding="utf-8") as file:
        file_writer = writer(file, delimiter=";", quotechar="'", quoting=QUOTE_MINIMAL)
        file_writer.writerow([run_nr, distance_measure, str(neighbour_count), descriptor_1, descriptor_2, str(weight), str(bin_count), "{:.2f}".format(guessing_acc), "{:.2f}".format(time_needed), str(image_set)])

    print("Step done.")

# Load images
tr1 = np.load("./train_images.npz", allow_pickle=True) # allow_pickle = "eingelegt" -> needed when loading object arrays (but not secure against erroneous or maliciously constructed data)
tr2 = np.load("./train_images_2.npz", allow_pickle=True)
tr3 = np.load("./train_images_3.npz", allow_pickle=True)
va1 = np.load("./val_images.npz", allow_pickle=True)
va2 = np.load("./val_images_2.npz", allow_pickle=True)
va3 = np.load("./val_images_3.npz", allow_pickle=True)

# Write header of CSV file (just necessary the first time)
#with open("Results.csv", mode="a", newline="\n", encoding="utf-8") as file:
#    file_writer = writer(file, delimiter=";", quotechar="'", quoting=QUOTE_MINIMAL)
#    file_writer.writerow(["run_nr", "distance_measure", "neighbour_count", "descriptor_1", "descriptor_2", "weight", "bin_count", "guessing_accuracy (%)", "time_needed (min)", "image_set"])

###############################################################################
# run(Parameter)-function with different settings:
#
# Parameter:         run_nr, image_set, distance_measure, neighbour_count, descriptor_1, descriptor_2, weight, bin_count
# image_set:         1, 2 or 3
# distance_measure:  euklid, intersect (doesn't work with "0" as a descriptor)
# neighbour_count:   number of compared suggestions for labels
# descriptor_x:      1d_histo, 3d_histo, std, mean, sobel, hog, 0
# weight:            gets multiplied with descriptor_2
# bin_count:         bins used in histogram
###############################################################################

#
# RUN 1: ERSTER TEST mit allen Image Sets jew. einmal
#
#run(1, "euklid", 8, "1d_histo", "std", 0, 8)
#run("euklid", 8, "3d_histo", "0", 0, 8)
#run("euklid", 8, "std", "0", 0, 0)
#run("euklid", 8, "mean", "0", 0, 0)
#run("euklid", 8, "sobel", "0", 0, 0)
#run("euklid", 8, "hog,4,8", "0", 0, 0)
#
#run("euklid", 4, "1d_histo", "0", 0, 8)
#run("euklid", 4, "3d_histo", "0", 0, 8)
#run("euklid", 4, "std", "0", 0, 0)
#run("euklid", 4, "mean", "0", 0, 0)
#run("euklid", 4, "sobel", "0", 0, 0)
#run("euklid", 4, "hog,4,8", "0", 0, 0)
#
#run("euklid", 3, "1d_histo", "0", 0, 8)
#run("euklid", 3, "3d_histo", "0", 0, 8)
#run("euklid", 3, "std", "0", 0, 0)
#run("euklid", 3, "mean", "0", 0, 0)
#run("euklid", 3, "sobel", "0", 0, 0)
#run("euklid", 3, "hog,4,8", "0", 0, 0)
#
#run("euklid", 2, "1d_histo", "0", 0, 8)
#run("euklid", 2, "3d_histo", "0", 0, 8)
#run("euklid", 2, "std", "0", 0, 0)
#run("euklid", 2, "mean", "0", 0, 0)
#run("euklid", 2, "sobel", "0", 0, 0)
#run("euklid", 2, "hog,4,8", "0", 0, 0)

#
# RUN 2: DA HISTO SO GUT LIEF
#
#run("euklid", 8, "1d_histo", "0", 0, 4)
#run("euklid", 8, "3d_histo", "0", 0, 4)
#run("euklid", 4, "1d_histo", "0", 0, 4)
#run("euklid", 4, "3d_histo", "0", 0, 4)
#run("euklid", 3, "1d_histo", "0", 0, 4)
#run("euklid", 3, "3d_histo", "0", 0, 4)
#run("euklid", 2, "1d_histo", "0", 0, 4)
#run("euklid", 2, "3d_histo", "0", 0, 4)
#
#run("euklid", 8, "1d_histo", "0", 0, 3)
#run("euklid", 8, "3d_histo", "0", 0, 3)
#run("euklid", 4, "1d_histo", "0", 0, 3)
#run("euklid", 4, "3d_histo", "0", 0, 3)
#run("euklid", 3, "1d_histo", "0", 0, 3)
#run("euklid", 3, "3d_histo", "0", 0, 3)
#run("euklid", 2, "1d_histo", "0", 0, 3)
#run("euklid", 2, "3d_histo", "0", 0, 3)

#
# RUN 3: HOG TESTEN
#
#run("euklid", 8, "hog,4,6", "0", 0, 0)
#run("euklid", 4, "hog,4,6", "0", 0, 0)
#run("euklid", 3, "hog,4,6", "0", 0, 0)
#run("euklid", 2, "hog,4,6", "0", 0, 0)
#run("euklid", 8, "hog,4,4", "0", 0, 0)
#run("euklid", 4, "hog,4,4", "0", 0, 0)

#
# RUN 4: MEHR NEIGHBOR COUNTS UND BINS (Histo)
#
#run("euklid", 8, "1d_histo", "0", 0, 5)
#run("euklid", 8, "3d_histo", "0", 0, 5)
#run("euklid", 4, "1d_histo", "0", 0, 5)
#run("euklid", 4, "3d_histo", "0", 0, 5)
#run("euklid", 3, "1d_histo", "0", 0, 5)
#run("euklid", 3, "3d_histo", "0", 0, 5)
#run("euklid", 2, "1d_histo", "0", 0, 5)
#run("euklid", 2, "3d_histo", "0", 0, 5)

#run("euklid", 8, "1d_histo", "0", 0, 6)
#run("euklid", 8, "3d_histo", "0", 0, 6)
#run("euklid", 4, "1d_histo", "0", 0, 6)
#run("euklid", 4, "3d_histo", "0", 0, 6)
#run("euklid", 3, "1d_histo", "0", 0, 6)
#run("euklid", 3, "3d_histo", "0", 0, 6)
#run("euklid", 2, "1d_histo", "0", 0, 6)
#run("euklid", 2, "3d_histo", "0", 0, 6)
#
#run("euklid", 8, "1d_histo", "0", 0, 7)
#run("euklid", 8, "3d_histo", "0", 0, 7)
#run("euklid", 4, "1d_histo", "0", 0, 7)
#run("euklid", 4, "3d_histo", "0", 0, 7)
#run("euklid", 3, "1d_histo", "0", 0, 7)
#run("euklid", 3, "3d_histo", "0", 0, 7)
#run("euklid", 2, "1d_histo", "0", 0, 7)
#run("euklid", 2, "3d_histo", "0", 0, 7)
#
#run("euklid", 8, "3d_histo", "0", 0, 2)

#
# RUN 5
#
#run("euklid", 2, "3d_histo", "mean", 0.9, 3)
#run("euklid", 2, "3d_histo", "mean", 0.7, 3)
#run("euklid", 2, "3d_histo", "mean", 0.5, 3)
#run("euklid", 2, "3d_histo", "mean", 0.3, 3)
#run("euklid", 2, "3d_histo", "mean", 0.1, 3)
#
#run("euklid", 2, "mean", "3d_histo", 0.9, 3)
#run("euklid", 2, "mean", "3d_histo", 0.7, 3)
#run("euklid", 2, "mean", "3d_histo", 0.5, 3)
#run("euklid", 2, "mean", "3d_histo", 0.3, 3)
#run("euklid", 2, "mean", "3d_histo", 0.1, 3)
#
#run("euklid", 6, "3d_histo", "std", 0.5, 3)
#run("euklid", 6, "std", "3d_histo", 0.5, 3)

#
# RUN 6
#
#run(1, "euklid", 8, "hog,6,8", "sobel", 0.3, 0)
#run(1, "euklid", 8, "hog,6,8", "sobel", 0.5, 0)
#run(1, "euklid", 8, "hog,6,8", "sobel", 0.7, 0)

#run(1, "euklid", 8, "sobel", "hog,6,8", 0.3, 0)
#run(1, "euklid", 8, "sobel", "hog,6,8", 0.5, 0)
#run(1, "euklid", 8, "sobel", "hog,6,8", 0.7, 0)
#
#run(1, "euklid", 4, "hog,6,8", "sobel", 0.3, 0)
#run(1, "euklid", 4, "hog,6,8", "sobel", 0.5, 0)
#run(1, "euklid", 4, "hog,6,8", "sobel", 0.7, 0)
#
#run(1, "euklid", 4, "sobel", "hog,6,8", 0.3, 0)
#run(1, "euklid", 4, "sobel", "hog,6,8", 0.5, 0)
#run(1, "euklid", 4, "sobel", "hog,6,8", 0.7, 0)
#
#run(1, "euklid", 2, "hog,6,8", "sobel", 0.3, 0)
#run(1, "euklid", 2, "hog,6,8", "sobel", 0.5, 0)
#run(1, "euklid", 2, "hog,6,8", "sobel", 0.7, 0)
#
#run(1, "euklid", 2, "sobel", "hog,6,8", 0.3, 0)
#run(1, "euklid", 2, "sobel", "hog,6,8", 0.5, 0)
#run(1, "euklid", 2, "sobel", "hog,6,8", 0.7, 0)
#
#
#run(2, "euklid", 8, "hog,6,8", "sobel", 0.3, 0)
#run(2, "euklid", 8, "hog,6,8", "sobel", 0.5, 0)
#run(2, "euklid", 8, "hog,6,8", "sobel", 0.7, 0)
#
#run(2, "euklid", 8, "sobel", "hog,6,8", 0.3, 0)
#run(2, "euklid", 8, "sobel", "hog,6,8", 0.5, 0)
#run(2, "euklid", 8, "sobel", "hog,6,8", 0.7, 0)
#
#run(2, "euklid", 4, "hog,6,8", "sobel", 0.3, 0)
#run(2, "euklid", 4, "hog,6,8", "sobel", 0.5, 0)
#run(2, "euklid", 4, "hog,6,8", "sobel", 0.7, 0)
#
#run(2, "euklid", 4, "sobel", "hog,6,8", 0.3, 0)
#run(2, "euklid", 4, "sobel", "hog,6,8", 0.5, 0)
#run(2, "euklid", 4, "sobel", "hog,6,8", 0.7, 0)
#
#run(2, "euklid", 2, "hog,6,8", "sobel", 0.3, 0)
#run(2, "euklid", 2, "hog,6,8", "sobel", 0.5, 0)
#run(2, "euklid", 2, "hog,6,8", "sobel", 0.7, 0)
#
#run(2, "euklid", 2, "sobel", "hog,6,8", 0.3, 0)
#run(2, "euklid", 2, "sobel", "hog,6,8", 0.5, 0)
#run(2, "euklid", 2, "sobel", "hog,6,8", 0.7, 0)
#
#
#run(3, "euklid", 8, "hog,6,8", "sobel", 0.3, 0)
#run(3, "euklid", 8, "hog,6,8", "sobel", 0.5, 0)
#run(3, "euklid", 8, "hog,6,8", "sobel", 0.7, 0)
#
#run(3, "euklid", 8, "sobel", "hog,6,8", 0.3, 0)
#run(3, "euklid", 8, "sobel", "hog,6,8", 0.5, 0)
#run(3, "euklid", 8, "sobel", "hog,6,8", 0.7, 0)
#
#run(3, "euklid", 4, "hog,6,8", "sobel", 0.3, 0)
#run(3, "euklid", 4, "hog,6,8", "sobel", 0.5, 0)
#run(3, "euklid", 4, "hog,6,8", "sobel", 0.7, 0)
#
#run(3, "euklid", 4, "sobel", "hog,6,8", 0.3, 0)
#run(3, "euklid", 4, "sobel", "hog,6,8", 0.5, 0)
#run(3, "euklid", 4, "sobel", "hog,6,8", 0.7, 0)
#
#run(3, "euklid", 2, "hog,6,8", "sobel", 0.3, 0)
#run(3, "euklid", 2, "hog,6,8", "sobel", 0.5, 0)
#run(3, "euklid", 2, "hog,6,8", "sobel", 0.7, 0)
#
#run(3, "euklid", 2, "sobel", "hog,6,8", 0.3, 0)
#run(3, "euklid", 2, "sobel", "hog,6,8", 0.5, 0)
#run(3, "euklid", 2, "sobel", "hog,6,8", 0.7, 0)

#
# RUN 7
# Only with image_set 1
#
#run("euklid", 4, "3d_histo", "mean", 0.9, 3)
#run("euklid", 4, "3d_histo", "mean", 0.7, 3)
#run("euklid", 4, "3d_histo", "mean", 0.5, 3)
#run("euklid", 4, "3d_histo", "mean", 0.3, 3)
#run("euklid", 4, "3d_histo", "mean", 0.1, 3)
#
#run("euklid", 5, "3d_histo", "mean", 0.9, 3)
#run("euklid", 5, "3d_histo", "mean", 0.7, 3)
#run("euklid", 5, "3d_histo", "mean", 0.5, 3)
#run("euklid", 5, "3d_histo", "mean", 0.3, 3)
#run("euklid", 5, "3d_histo", "mean", 0.1, 3)
#
#run("euklid", 6, "3d_histo", "mean", 0.9, 3)
#run("euklid", 6, "3d_histo", "mean", 0.7, 3)
#run("euklid", 6, "3d_histo", "mean", 0.5, 3)
#run("euklid", 6, "3d_histo", "mean", 0.3, 3)
#run("euklid", 6, "3d_histo", "mean", 0.1, 3)
#
#run("euklid", 7, "3d_histo", "mean", 0.9, 3)
#run("euklid", 7, "3d_histo", "mean", 0.7, 3)
#run("euklid", 7, "3d_histo", "mean", 0.5, 3)
#run("euklid", 7, "3d_histo", "mean", 0.3, 3)
#run("euklid", 7, "3d_histo", "mean", 0.1, 3)
#
#run("euklid", 8, "3d_histo", "mean", 0.9, 3)
#run("euklid", 8, "3d_histo", "mean", 0.7, 3)
#run("euklid", 8, "3d_histo", "mean", 0.5, 3)
#run("euklid", 8, "3d_histo", "mean", 0.3, 3)
#run("euklid", 8, "3d_histo", "mean", 0.1, 3)
#
#run("euklid", 4, "mean", "3d_histo", 0.9, 3)
#run("euklid", 4, "mean", "3d_histo", 0.7, 3)
#run("euklid", 4, "mean", "3d_histo", 0.5, 3)
#run("euklid", 4, "mean", "3d_histo", 0.3, 3)
#run("euklid", 4, "mean", "3d_histo", 0.1, 3)
#
#run("euklid", 5, "mean", "3d_histo", 0.9, 3)
#run("euklid", 5, "mean", "3d_histo", 0.7, 3)
#run("euklid", 5, "mean", "3d_histo", 0.5, 3)
#run("euklid", 5, "mean", "3d_histo", 0.3, 3)
#run("euklid", 5, "mean", "3d_histo", 0.1, 3)
#
#run("euklid", 6, "mean", "3d_histo", 0.9, 3)
#run("euklid", 6, "mean", "3d_histo", 0.7, 3)
#run("euklid", 6, "mean", "3d_histo", 0.5, 3)
#run("euklid", 6, "mean", "3d_histo", 0.3, 3)
#run("euklid", 6, "mean", "3d_histo", 0.1, 3)
#
#run("euklid", 7, "mean", "3d_histo", 0.9, 3)
#run("euklid", 7, "mean", "3d_histo", 0.7, 3)
#run("euklid", 7, "mean", "3d_histo", 0.5, 3)
#run("euklid", 7, "mean", "3d_histo", 0.3, 3)
#run("euklid", 7, "mean", "3d_histo", 0.1, 3)
#
#run("euklid", 8, "mean", "3d_histo", 0.9, 3)
#run("euklid", 8, "mean", "3d_histo", 0.7, 3)
#run("euklid", 8, "mean", "3d_histo", 0.5, 3)
#run("euklid", 8, "mean", "3d_histo", 0.3, 3)
#run("euklid", 8, "mean", "3d_histo", 0.1, 3)   
    
#
# RUN 9 (goes through all 3 image sets)
# 3d_histo mal mit mehr kombinieren
#run_nr = 9
#
#for i in range(3):
#    run(run_nr, i+1, "euklid", 2, "3d_histo", "sobel", 0.3, 5)
#    run(run_nr, i+1, "euklid", 2, "3d_histo", "sobel", 0.5, 5)
#    run(run_nr, i+1, "euklid", 2, "3d_histo", "sobel", 0.7, 5)
#    run(run_nr, i+1, "euklid", 2, "sobel", "3d_histo", 0.3, 5)
#    run(run_nr, i+1, "euklid", 2, "sobel", "3d_histo", 0.5, 5)
#    run(run_nr, i+1, "euklid", 2, "sobel", "3d_histo", 0.7, 5)  
#    run(run_nr, i+1, "euklid", 2, "3d_histo", "hog,4,8", 0.5, 5)
#    run(run_nr, i+1, "euklid", 2, "hog,4,8", "3d_histo", 0.5, 5)
#    
#    run(run_nr, i+1, "euklid", 3, "3d_histo", "sobel", 0.3, 5)
#    run(run_nr, i+1, "euklid", 3, "3d_histo", "sobel", 0.5, 5)
#    run(run_nr, i+1, "euklid", 3, "3d_histo", "sobel", 0.7, 5)
#    run(run_nr, i+1, "euklid", 3, "sobel", "3d_histo", 0.3, 5)
#    run(run_nr, i+1, "euklid", 3, "sobel", "3d_histo", 0.5, 5)
#    run(run_nr, i+1, "euklid", 3, "sobel", "3d_histo", 0.7, 5)  
#    run(run_nr, i+1, "euklid", 3, "3d_histo", "hog,4,8", 0.5, 5)
#    run(run_nr, i+1, "euklid", 3, "hog,4,8", "3d_histo", 0.5, 5)
#    
#    run(run_nr, i+1, "euklid", 7, "3d_histo", "sobel", 0.3, 5)
#    run(run_nr, i+1, "euklid", 7, "3d_histo", "sobel", 0.5, 5)
#    run(run_nr, i+1, "euklid", 7, "3d_histo", "sobel", 0.7, 5)
#    run(run_nr, i+1, "euklid", 7, "sobel", "3d_histo", 0.3, 5)
#    run(run_nr, i+1, "euklid", 7, "sobel", "3d_histo", 0.5, 5)
#    run(run_nr, i+1, "euklid", 7, "sobel", "3d_histo", 0.7, 5)  
#    run(run_nr, i+1, "euklid", 7, "3d_histo", "hog,4,8", 0.5, 5)
#    run(run_nr, i+1, "euklid", 7, "hog,4,8", "3d_histo", 0.5, 5)
#    
#    run(run_nr, i+1, "euklid", 8, "3d_histo", "sobel", 0.3, 5)
#    run(run_nr, i+1, "euklid", 8, "3d_histo", "sobel", 0.5, 5)
#    run(run_nr, i+1, "euklid", 8, "3d_histo", "sobel", 0.7, 5)
#    run(run_nr, i+1, "euklid", 8, "sobel", "3d_histo", 0.3, 5)
#    run(run_nr, i+1, "euklid", 8, "sobel", "3d_histo", 0.5, 5)
#    run(run_nr, i+1, "euklid", 8, "sobel", "3d_histo", 0.7, 5)  
#    run(run_nr, i+1, "euklid", 8, "3d_histo", "hog,4,8", 0.5, 5)
#    run(run_nr, i+1, "euklid", 8, "hog,4,8", "3d_histo", 0.5, 5)


#
# RUN 9 (goes through all 3 image sets)
# 3d_histo mit anderem bin count und erfolgreichsten neigbour count
run_nr = 9
#
for i in range(3):
    if i+1 != 1:
        run(run_nr, i+1, "euklid", 2, "3d_histo", "sobel", 0.3, 3)
        run(run_nr, i+1, "euklid", 2, "3d_histo", "sobel", 0.5, 3)  
    run(run_nr, i+1, "euklid", 2, "3d_histo", "sobel", 0.7, 3)
    run(run_nr, i+1, "euklid", 2, "3d_histo", "hog,4,8", 0.5, 3)
    
    run(run_nr, i+1, "euklid", 3, "3d_histo", "sobel", 0.3, 3)
    run(run_nr, i+1, "euklid", 3, "3d_histo", "sobel", 0.5, 3)
    run(run_nr, i+1, "euklid", 3, "3d_histo", "sobel", 0.7, 3)
    run(run_nr, i+1, "euklid", 3, "3d_histo", "hog,4,8", 0.5, 3)
    

#
# RUN 10
# INTERSECT VERGLEICHE
# Only with image Set 1
#
#run(1, "intersect", 8, "1d_histo", "std", 0, 8)
#run(1, "euklid", 8, "1d_histo", "std", 0, 8)
#run(1, "intersect", 7, "3d_histo", "mean", 0.3, 3) #-->mit euklid bei 38.75%, mit intersect bei?

#
# RUN 11: wie RUN 1 nur mit intersect
#run_nr = 10
#
#run(run_nr, 1, "intersect", 8, "1d_histo", "std", 0, 8)
#run(run_nr, 1, "intersect", 8, "3d_histo", "std", 0, 8)
#run(run_nr, 1, "intersect", 8, "std", "std", 0, 0)
#run(run_nr, 1, "intersect", 8, "mean", "std", 0, 0)
#run(run_nr, 1, "intersect", 8, "sobel", "std", 0, 0)
#run(run_nr, 1, "intersect", 8, "hog,4,8", "std", 0, 0)
#run(run_nr, 1, "intersect", 4, "1d_histo", "std", 0, 8)
#run(run_nr, 1, "intersect", 4, "3d_histo", "std", 0, 8)
#run(run_nr, 1, "intersect", 4, "std", "std", 0, 0)
#run(run_nr, 1, "intersect", 4, "mean", "std", 0, 0)
#run(run_nr, 1, "intersect", 4, "sobel", "std", 0, 0)
#run(run_nr, 1, "intersect", 4, "hog,4,8", "std", 0, 0)
#run(run_nr, 1, "intersect", 3, "1d_histo", "std", 0, 8)
#run(run_nr, 1, "intersect", 3, "3d_histo", "std", 0, 8)
#run(run_nr, 1, "intersect", 3, "std", "std", 0, 0)
#run(run_nr, 1, "intersect", 3, "mean", "std", 0, 0)
#run(run_nr, 1, "intersect", 3, "sobel", "std", 0, 0)
#run(run_nr, 1, "intersect", 3, "hog,4,8", "std", 0, 0)
#run(run_nr, 1, "intersect", 2, "1d_histo", "std", 0, 8)
#run(run_nr, 1, "intersect", 2, "3d_histo", "std", 0, 8)
#run(run_nr, 1, "intersect", 2, "std", "std", 0, 0)
#run(run_nr, 1, "intersect", 2, "mean", "std", 0, 0)
#run(run_nr, 1, "intersect", 2, "sobel", "std", 0, 0)
#run(run_nr, 1, "intersect", 2, "hog,4,8", "std", 0, 0)

#run(run_nr, 2, "intersect", 8, "1d_histo", "std", 0, 8)
#run(run_nr, 2, "intersect", 8, "3d_histo", "std", 0, 8)
#run(run_nr, 2, "intersect", 8, "std", "std", 0, 0)
#run(run_nr, 2, "intersect", 8, "mean", "std", 0, 0)
#run(run_nr, 2, "intersect", 8, "sobel", "std", 0, 0)
#run(run_nr, 2, "intersect", 8, "hog,4,8", "std", 0, 0)
#run(run_nr, 2, "intersect", 4, "1d_histo", "std", 0, 8)
#run(run_nr, 2, "intersect", 4, "3d_histo", "std", 0, 8)
#run(run_nr, 2, "intersect", 4, "std", "std", 0, 0)
#run(run_nr, 2, "intersect", 4, "mean", "std", 0, 0)
#run(run_nr, 2, "intersect", 4, "sobel", "std", 0, 0)
#run(run_nr, 2, "intersect", 4, "hog,4,8", "std", 0, 0)
#run(run_nr, 2, "intersect", 3, "1d_histo", "std", 0, 8)
#run(run_nr, 2, "intersect", 3, "3d_histo", "std", 0, 8)
#run(run_nr, 2, "intersect", 3, "std", "std", 0, 0)
#run(run_nr, 2, "intersect", 3, "mean", "std", 0, 0)
#run(run_nr, 2, "intersect", 3, "sobel", "std", 0, 0)
#run(run_nr, 2, "intersect", 3, "hog,4,8", "std", 0, 0)
#run(run_nr, 2, "intersect", 2, "1d_histo", "std", 0, 8)
#run(run_nr, 2, "intersect", 2, "3d_histo", "std", 0, 8)
#run(run_nr, 2, "intersect", 2, "std", "std", 0, 0)
#run(run_nr, 2, "intersect", 2, "mean", "std", 0, 0)
#run(run_nr, 2, "intersect", 2, "sobel", "std", 0, 0)
#run(run_nr, 2, "intersect", 2, "hog,4,8", "std", 0, 0)
#
#run(run_nr, 3, "intersect", 8, "1d_histo", "std", 0, 8)
#run(run_nr, 3, "intersect", 8, "3d_histo", "std", 0, 8)
#run(run_nr, 3, "intersect", 8, "std", "std", 0, 0)
#run(run_nr, 3, "intersect", 8, "mean", "std", 0, 0)
#run(run_nr, 3, "intersect", 8, "sobel", "std", 0, 0)
#run(run_nr, 3, "intersect", 8, "hog,4,8", "std", 0, 0)
#run(run_nr, 3, "intersect", 4, "1d_histo", "std", 0, 8)
#run(run_nr, 3, "intersect", 4, "3d_histo", "std", 0, 8)
#run(run_nr, 3, "intersect", 4, "std", "std", 0, 0)
#run(run_nr, 3, "intersect", 4, "mean", "std", 0, 0)
#run(run_nr, 3, "intersect", 4, "sobel", "std", 0, 0)
#run(run_nr, 3, "intersect", 4, "hog,4,8", "std", 0, 0)
#run(run_nr, 3, "intersect", 3, "1d_histo", "std", 0, 8)
#run(run_nr, 3, "intersect", 3, "3d_histo", "std", 0, 8)
#run(run_nr, 3, "intersect", 3, "std", "std", 0, 0)
#run(run_nr, 3, "intersect", 3, "mean", "std", 0, 0)
#run(run_nr, 3, "intersect", 3, "sobel", "std", 0, 0)
#run(run_nr, 3, "intersect", 3, "hog,4,8", "std", 0, 0)
#run(run_nr, 3, "intersect", 2, "1d_histo", "std", 0, 8)
#run(run_nr, 3, "intersect", 2, "3d_histo", "std", 0, 8)
#run(run_nr, 3, "intersect", 2, "std", "std", 0, 0)
#run(run_nr, 3, "intersect", 2, "mean", "std", 0, 0)
#run(run_nr, 3, "intersect", 2, "sobel", "std", 0, 0)
#run(run_nr, 3, "intersect", 2, "hog,4,8", "std", 0, 0)

print("\nDone!")
