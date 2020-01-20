# -*- coding: utf-8 -*-

#import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
#from skimage import data, exposure
#from skimage.io import imread, imsave
from sklearn.utils import shuffle
from skimage.filters import sobel_h, sobel_v
from skimage.color import rgb2gray
import timeit


# Trainings-Bilder laden und vorbereiten
tr = np.load('./train_images.npz', allow_pickle=True)
tr_images = tr['data']
tr_labels = tr['labels']

# Validierungs-Bilder laden und vorbereiten
va = np.load('./val_images.npz', allow_pickle=True)
va_images = va['data']
va_labels = va['labels']

# Shuffle arrays
tr_shuffled_images, tr_shuffled_labels = shuffle(tr_images, tr_labels)
va_shuffled_images, va_shuffled_labels = shuffle(va_images, va_labels)


# Calculate estimated labels and write to file
def calculate_estimated_labels(p_bin_count, p_neighbour_count, p_dist_ma, p_descriptor_1, p_descriptor_2, p_weight):
    #open results_file
    results_file = open("Results.html", "a")
    
    def rgb_img_to_1d_histo(img):
    	img = img.ravel() # ravel => returns flattend 1d-Array
    	return np.histogram(img, bins = bin_count, range=(0,256))
    
    def rgb_img_to_3d_histo(img):
    	img = img.reshape(img.shape[0] * img.shape[1], 3)
    	return np.histogramdd(img, bins = [bin_count,bin_count,bin_count], range=((0,256),(0,256),(0,256)))
    
    def eukl_dist(x,y):
    	return np.sqrt(np.sum((x-y)**2))
    
    def intersection(hist_1, hist_2):
        minima = np.minimum(hist_1, hist_2)
        intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
        return intersection
    
    def rgb_img_mean(img): # der Farbmittelwert eines Bildes
    	return np.mean(img, axis = (0, 1, 2))
    
    def indices_of_n_smallest_values(neighbors, num_list): # gibt die indizes der n kleinsten elemente des arrays zurück
    	return np.array(num_list).argsort()[:neighbors]
    
    def indices_of_n_biggest_values(n, num_list): # gibt die indizes der n größten elemente des arrays zurück
    	return np.array(num_list).argsort()[-n:][::-1]
    
    def elements_for_index_list(index_list, elements_list): # gibt fuer eine liste an indizes und eine an elementen die elemente der indizes zurueck
    	return list(map(lambda x: elements_list[x], index_list))
    
    def most_common_occurrence(element_list): # gibt das am oeften auftretende element zurueck
    	return max(element_list,key=element_list.count)
    
    def calculate_combined_weighted_distanze(img_1, img_2): # nimmt die euklidische distanz von dem mittelwert und histogrammen und summiert sie gewichtet auf
    	# Beladen der ersten beiden Merkmale
        if descriptor_1 == '1d_histo':
            deskr_1_img_1 = rgb_img_to_1d_histo(img_1)[0]
            deskr_1_img_2 = rgb_img_to_1d_histo(img_2)[0]
        elif descriptor_1 == '3d_histo':
            deskr_1_img_1 = rgb_img_to_3d_histo(img_1)[0]
            deskr_1_img_2 = rgb_img_to_3d_histo(img_2)[0]
        elif descriptor_1 == 'std':
            deskr_1_img_1 = np.std(img_1)
            deskr_1_img_2 = np.std(img_2)
        elif descriptor_1 == 'mean':
            deskr_1_img_1 = rgb_img_mean(img_1)
            deskr_1_img_2 = rgb_img_mean(img_2)
        elif descriptor_1 == '0':
            deskr_1_img_1 = 0
            deskr_1_img_2 = 0
        elif descriptor_1 == 'sobel':
            ### MUSS GRAUSTUFEN BILD SEIN !!!!!!!!!!!!!
            sobelH_1 = sobel_h(rgb2gray(img_1)) #horizontale Kanten/Helligkeitsunterschiede finden mit dem Sobelfilter
            sobelV_1 = sobel_v(rgb2gray(img_1)) #vertikale Kanten/Helligkeitsunterschiede finden mit dem Sobelfilter
            deskr_1_img_1 = np.sqrt(sobelH_1**2 + sobelV_1**2) #gradientStaerke: die Staerke des Gradienten berechnet sich durch die Laenge des Vektors, der aus den beiden Einzelteilen sobelH und sobelV entsteht
            sobelH_2 = sobel_h(rgb2gray(img_2))
            sobelV_2 = sobel_v(rgb2gray(img_2))
            deskr_1_img_2 = np.sqrt(sobelH_2**2 + sobelV_2**2)
            #plt.imshow(deskr_1_img_2)
            #plt.show()
        elif descriptor_1 == 'hog':
            deskr_1_img_1 = hog(img_1, orientations=4, pixels_per_cell=(8, 8))
            deskr_1_img_2 = hog(img_2, orientations=4, pixels_per_cell=(8, 8))
            # So gibt man noch ein Bild dazu aus:
            #deskr_1_img_1, theBild = hog(img_1, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
            #plt.imshow(theBild)
            #plt.show()
    
    
    	# Beladen der zweiten beiden Merkmale
        if descriptor_2 == '1d_histo':
            deskr_2_img_1 = rgb_img_to_1d_histo(img_1)[0]
            deskr_2_img_2 = rgb_img_to_1d_histo(img_2)[0]
        elif descriptor_2 == '3d_histo':
            deskr_2_img_1 = rgb_img_to_3d_histo(img_1)[0]
            deskr_2_img_2 = rgb_img_to_3d_histo(img_2)[0]
        elif descriptor_2 == 'std': #Standardabweichung entlang der angegebenen Achse
            deskr_2_img_1 = np.std(img_1)
            deskr_2_img_2 = np.std(img_2)
        elif descriptor_2 == 'mean': #Mittelwert entlang der angegebenen Achse
            deskr_2_img_1 = rgb_img_mean(img_1)
            deskr_2_img_2 = rgb_img_mean(img_2)
        elif descriptor_2 == '0':
            deskr_2_img_1 = 0
            deskr_2_img_2 = 0
        elif descriptor_2 == 'sobel': #Kanten finden
            ### MUSS GRAUSTUFEN BILD SEIN !!!!!!!!!!!!!
            sobelH_1 = sobel_h(rgb2gray(img_1)) #horizontale Kanten/Helligkeitsunterschiede finden mit dem Sobelfilter
            sobelV_1 = sobel_v(rgb2gray(img_1)) #vertikale Kanten/Helligkeitsunterschiede finden mit dem Sobelfilter
            deskr_2_img_1 = np.sqrt(sobelH_1**2 + sobelV_1**2) #gradientStaerke: die Staerke des Gradienten berechnet sich durch die Laenge des Vektors, der aus den beiden Einzelteilen sobelH und sobelV entsteht
            sobelH_2 = sobel_h(rgb2gray(img_2))
            sobelV_2 = sobel_v(rgb2gray(img_2))
            deskr_2_img_2 = np.sqrt(sobelH_2**2 + sobelV_2**2)
        elif descriptor_2 == 'hog': #Histogramm über Gradientenorientierung 
            deskr_2_img_1 = hog(img_1, orientations=4, pixels_per_cell=(8, 8))
            deskr_2_img_2 = hog(img_2, orientations=4, pixels_per_cell=(8, 8))
    
    
        if dist_ma == 'euklid':
            distance_1 = eukl_dist(deskr_1_img_1, deskr_1_img_2)
            distance_2 = eukl_dist(deskr_2_img_1, deskr_2_img_2)
        elif dist_ma == 'intersect':
            distance_1 = intersection(deskr_1_img_1, deskr_1_img_2)
            distance_2 = intersection(deskr_2_img_1, deskr_2_img_2)
    
        return distance_1 + weight * distance_2
    
    def rgb_img_n_nearest_neighbour(va_imgs, tr_imgs): # brechnet durch die n naechsten nachbarn das warcheinliche label
        print("\nBitte warten, am rechnen...")
        waitLength = len(va_imgs)
        est_labels_list = []
        for i, va_img in enumerate(va_imgs):
            #print("1. for loop")
            distances = []
            for tr_img in tr_imgs:
                #print("2. for loop")
                distance = calculate_combined_weighted_distanze(va_img, tr_img)
                distances.append(distance)
            index_list = indices_of_n_smallest_values(neighbour_count, distances)
            #print("index_list: " + str(index_list))
            n_closest_labels = elements_for_index_list(index_list, tr_shuffled_labels)
            #print("n_closest_labels: " + str(n_closest_labels))
            #print("n_closest_labels: " + str(n_closest_labels))
            est_label = most_common_occurrence(n_closest_labels)
            #print("est_label: " + str(est_label))
            est_labels_list.append(est_label)
            print(str(i + 1) + " von " + str(waitLength))
        return est_labels_list
    
    def guessing_accuracy(est_labels, val_labels): # Wirft auf der console infos aus und retuned genauigkeit in %
        count = 0
        for index in range(len(est_labels)):
            if est_labels[index] == val_labels[index]:
                count += 1
        return str(count / len(va_images) * 100) + "%"
    
    # Params
    bin_count = p_bin_count #for histogram
    neighbour_count = p_neighbour_count #anzahl der verglichenen vorschläge für labels
    weight = p_weight #for the descriptor_2, 0-1
    dist_ma = p_dist_ma # distanzmas ist entweder euklid oder intersect
    descriptor_1 = p_descriptor_1 # diskriptoren sind: 1d_histo, 3d_histo, std, mean, 0, sobel, hog
    descriptor_2 = p_descriptor_2    #for time measurement
    time_start = 0
    time_stop = 0
    
    #Zeitmessung starten
    time_start = timeit.default_timer()
    
    # Ausfuehren der Auswertung
    estimated_labels = rgb_img_n_nearest_neighbour(va_shuffled_images, tr_shuffled_images)
    
    #Zeitmessung stoppen
    time_stop = timeit.default_timer()
    needed_time = (time_stop - time_start) / 60 #in Minuten
    
    #Trefferquote
    guessing_accuracy = guessing_accuracy(estimated_labels, va_shuffled_labels)
    
    # Print Settings
    results_file.write("<table>\n<tr><td>Distanzmaß</td><td>Neighbour Count</td><td>Deskriptor 1</td><td>Deskriptor 2 / Gewichtung</td><td>Bin Count</td><td>Berechnungszeit</td><td>Trefferquote</td></tr>\n<tr><td>" + dist_ma + "</td><td>" + str(neighbour_count) + "</td><td>" + descriptor_1 + "</td><td>" + descriptor_2 + " / " + str(weight) + "</td><td>" + str(bin_count) + "</td><td>" + str(needed_time) + "min</td><td>" + guessing_accuracy + "</td></tr>\n</table>\n\n")
    results_file.close()

######################
# Parameter: bin_count, neighbour_count, dist_ma, descriptor_1, descriptor_2, weight)
# desriptor_x = 1d_histo, 3d_histo, std, mean, sobel, hog, 0
calculate_estimated_labels(8, 8, 'euklid', 'std', 'sobel', 0)
calculate_estimated_labels(8, 8, 'euklid', 'mean', 'sobel', 0)
