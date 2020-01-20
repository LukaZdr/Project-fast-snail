#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import
import numpy as np
np.random.seed(123) # um die Gewichte immer gleich zufaellig zu initialisieren
import tensorflow as tf
tf.set_random_seed(123) # um die Gewichte immer gleich zufaellig zu initialisieren
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD

# Laden der Daten

# Features Berechnen
# moegliche features die angemessen sind(wikipedia)
# @COLOR: 
  # Dominant Color Descriptor (DCD)
  # Scalable Color Descriptor (SCD)
  # Color Structure Descriptor (CSD)
  # Color Layout Descriptor (CLD)
  # Group of frame (GoF) or Group-of-pictures (GoP)
# SHAPE:
  # Regi`on-based Shape Descriptor (RSD)
  # Contour-based Shape Descriptor (CSD)
  # 3-D `Shape Descriptor (3-D SD)
  # Histogram of oriented gradients (HOG)


# def euklDist(x,y):
#   return np.sqrt(np.sum((x-y)**2))

def labels_to_y_train(labels):
	new_labels = []
	for label in labels:
		if label == 1: new_label = 0
		if label == 2: new_label = 1
		if label == 3: new_label = 2
		if label == 4: new_label = 3
		new_labels.append(new_label)
	return np_utils.to_categorical(new_labels, 4)


# Aufgabe 1
def get_rgb_mws_stds(imgs): # kanalweiser Mittelwert sowie Standardabweichung
	x_train = []
	for index in range(len(imgs)):
		img = imgs[index,:,:,:]
		img_vals = []
		img_vals.append(np.mean(img[:,:,0]))
		img_vals.append(np.mean(img[:,:,1]))
		img_vals.append(np.mean(img[:,:,2]))
		img_vals.append(np.std(img[:,:,0]))
		img_vals.append(np.std(img[:,:,1]))
		img_vals.append(np.std(img[:,:,2]))
		x_train.append(img_vals)
	return np.array(x_train)

# Aufgabe 3
tr_data = np.load('./train_images.npz')
tr_imgs = tr_data['data']
tr_labels = tr_data['labels']
y_train = labels_to_y_train(tr_labels)

va_data = np.load('./val_images.npz')
vl_imgs = va_data['data']
vl_labels = va_data['labels']
y_test = labels_to_y_train(vl_labels)

#Trainingsdaten berechnen:
x_train = get_rgb_mws_stds(tr_imgs)

#Vlidierungsdaten berechnen:
vl_rgb_mws_stds = get_rgb_mws_stds(vl_imgs)

# est_label = []

# for i in range(len(vl_imgs)):
#     dist = []
#     for i2 in range(len(tr_imgs)):
#         dist.append(euklDist(vl_imgs_mean[i],tr_imgs_mean[i2]))
#     est_label.append(trLabels[dist.index(np.min(dist))])
# count = 0
# for ind in range(len(est_label)):
#     if est_label[ind] == vlLabels[ind]: count += 1

# print("Treffer:")
# print(count)
# print(count/len(vl_imgs)*100, "%")

# Aufgabe 4
model = Sequential()
model.add(Dense(20, activation='relu', name='fc1',input_shape=(6,)))
model.add(Dense(20, activation='relu', name='fc2'))
model.add(Dense(20, activation='relu', name='fc3'))
model.add(Dense(4, activation='softmax'))

# Aufgabe 5
model.compile(loss='categorical_crossentropy',
          		optimizer=SGD(lr=0.000005, momentum=0.9),
							metrics=['accuracy'])

# Aufgabe 6
model.fit(x_train, y_train, batch_size=1, epochs=500, verbose=1)

# Aufgabe 7
# Epoch 500/500
# 60/60 [==============================] - 0s 1ms/step - loss: 0.8234 - accuracy: 0.5333

"""die Genauigkeit des Systems, mit 2 Denselayern mit jeweils 8 Neuronen, hat sich nur
minimal verbessert. Deht man aber ab der Zahl der Denselayer oder Neuronen pro Layer hoch
so verbessert sich die genauigkeit enorm."""