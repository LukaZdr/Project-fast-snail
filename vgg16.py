#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import
import numpy as np
np.random.seed(123) # um die Gewichte immer gleich zufaellig zu initialisieren
import tensorflow as tf
tf.compat.v1.set_random_seed(123) # um die Gewichte immer gleich zufaellig zu initialisieren
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16



def labels_to_y_train(labels):
	new_labels = []
	for label in labels:
		if label == 1: new_label = 0
		if label == 2: new_label = 1
		if label == 3: new_label = 2
		if label == 4: new_label = 3
		new_labels.append(new_label)
	return np_utils.to_categorical(new_labels, 4)

# Laden der Daten
tr_data = np.load('./train_images.npz')
tr_imgs = tr_data['data']
tr_labels = tr_data['labels']
y_train = labels_to_y_train(tr_labels)

va_data = np.load('./val_images.npz')
vl_imgs = va_data['data']
vl_labels = va_data['labels']
y_test = labels_to_y_train(vl_labels)

vgg16_model = VGG16()

model = Sequential()
for layer in vgg16_model.layers:
	model.add(layer)

model.layers.pop()

for layer in model.layers:
	layer.trainable = False

model.add(Dense(4, activation='softmax'))

model.summary()


# model.compile(loss='categorical_crossentropy',
#           		optimizer=SGD(lr=0.000005, momentum=0.9),
# 							metrics=['accuracy'])

# model.fit(tr_imgs, y_train, batch_size=1, epochs=4, verbose=1)

# score = model.evaluate(vl_imgs, y_test, verbose=1)

# print(score)
