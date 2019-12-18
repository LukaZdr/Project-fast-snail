# -*- coding: utf-8 -*-

import numpy as np

def rgb_img_to_1d_histo(img):
	img = img.ravel() # ravel => returns flattend 1d-Array
	return np.histogram(img, bins = bin_count, range=(0,256))

def rgb_img_to_3d_histo(img):
	img =  img.reshape((img.shape[0]*img.shape[1],3))
	return np.histogramdd(img, bins = [bin_count,bin_count,bin_count], range=((0,256),(0,256),(0,256)))

def eukl_dist(x,y):
		return np.sqrt(np.sum((x-y)**2))

def intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

def rgb_img_mean(img): # der Farbmittelwert eines Bildes
	return np.mean(img, axis = (0,1,2))

def indices_of_n_smallest_values(n, num_list): # gibt die indizes der n kleinsten elemente des arrays zurück
	return np.array(num_list).argsort()[:n]

def indices_of_n_biggest_values(n, num_list): # gibt die indizes der n größten elemente des arrays zurück
	return np.array(num_list).argsort()[-n:][::-1]

def elements_for_index_list(index_list, elements_list): # gibt fuer eine liste an indizes und eine an elementen die elemente der indizes zurueck
	return list(map(lambda x: elements_list[x], index_list))

def most_common_occurrence(element_list): # gibt das am oeften auftretende element zurueck
	return max(element_list,key=element_list.count)

def calculate_combined_weighted_distanze(image_1, image_2): # nimmt die euklidische distanz von dem mittelwert und histogrammen und summiert sie gewichtet auf
	# Beladen der ersten beiden Merkmale
	if descriptor_1 == '1d_histo':
		deskr_1_img_1 = rgb_img_to_1d_histo(image_1)[0]
		deskr_1_img_2 = rgb_img_to_1d_histo(image_2)[0]
	elif descriptor_1 == '3d_histo':
		deskr_1_img_1 = rgb_img_to_3d_histo(image_1)[0]
		deskr_1_img_2 = rgb_img_to_3d_histo(image_2)[0]
	elif descriptor_1 == 'std':
		deskr_1_img_1 = np.std(image_1)
		deskr_1_img_2 = np.std(image_2)
	elif descriptor_1 == 'mean':
		deskr_1_img_1 = rgb_img_mean(image_1)
		deskr_1_img_2 = rgb_img_mean(image_2)
	elif descriptor_1 == '0':
		deskr_1_img_1 = 0
		deskr_1_img_2 = 0

	# Beladen der zweiten beiden Merkmale
	if descriptor_2 == '1d_histo':
		deskr_2_img_1 = rgb_img_to_1d_histo(image_1)[0]
		deskr_2_img_2 = rgb_img_to_1d_histo(image_2)[0]
	elif descriptor_2 == '3d_histo':
		deskr_2_img_1 = rgb_img_to_3d_histo(image_1)[0]
		deskr_2_img_2 = rgb_img_to_3d_histo(image_2)[0]
	elif descriptor_2 == 'std':
		deskr_2_img_1 = np.std(image_1)
		deskr_2_img_2 = np.std(image_2)
	elif descriptor_2 == 'mean':
		deskr_2_img_1 = rgb_img_mean(image_1)
		deskr_2_img_2 = rgb_img_mean(image_2)
	elif descriptor_2 == '0':
		deskr_2_img_1 = 0
		deskr_2_img_2 = 0

	if dist_ma == 'euklid':
		distance_1 = eukl_dist(deskr_1_img_1, deskr_1_img_2)
		distance_2 = eukl_dist(deskr_2_img_1, deskr_2_img_2)
	elif dist_ma == 'intersect':
		distance_1 = intersection(deskr_1_img_1, deskr_1_img_2)
		distance_2 = intersection(deskr_2_img_1, deskr_2_img_2)

	return distance_1 + weight * distance_2

def rgb_img_n_nearest_neighbour(va_imgs, tr_imgs): # brechnet durch die n naechsten nachbarn das warcheinliche label
    print("processing image")
    est_labels_list = []
    for va_img in va_imgs:
        distances = []
        for tr_img in tr_imgs:
            distance = calculate_combined_weighted_distanze(va_img, tr_img)
            distances.append(distance)
        index_list = indices_of_n_smallest_values(neighbour_count, distances)
        n_closest_labels = elements_for_index_list(index_list, tr_labels)
        est_label = most_common_occurrence(n_closest_labels)
        est_labels_list.append(est_label)
    return est_labels_list

def guessing_accuracy(est_labels, va_labels): # Wirft auf der console infos aus und retuned genauigkeit in %
	count = 0
	for ind in range(len(est_labels)):
			if est_labels[ind] == va_labels[ind]: count += 1
	return count/len(va_imgs)*100

# Params
bin_count = 3
neighbour_count = 3
weight = 1
dist_ma = 'euklid' # distanzmas ist entweder euklid oder intersect
descriptor_1 = '3d_histo' # diskriptoren sind: 1d_histo, 3d_histo, std, mean
descriptor_2 = '0'

# Bilder laden und vorbereiten
d = np.load('./image_library.npz', allow_pickle=True)
tr_imgs = d['data']
tr_labels = d['labels']

# Bilder laden und vorbereiten
v = np.load('./image_library.npz', allow_pickle=True)
va_imgs = v['data']
va_labels = v['labels']

#Ausfuehren der Auswertung
est_labels = rgb_img_n_nearest_neighbour(va_imgs, tr_imgs)

# Statistik
print(guessing_accuracy(est_labels, va_labels))
