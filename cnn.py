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

# Laden der Daten

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

model = Sequential()
model.add(Dense(20, activation='relu', name='fc1',input_shape=(6,)))
model.add(Dense(20, activation='relu', name='fc2'))
model.add(Dense(20, activation='relu', name='fc3'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',
          		optimizer=SGD(lr=0.000005, momentum=0.9),
							metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=1, epochs=500, verbose=1)

score = model.evaluate(vl_rgb_mws_stds, y_test, verbose=1)

print(score)
