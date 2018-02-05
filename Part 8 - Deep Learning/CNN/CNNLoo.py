#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:37:24 2018

@author: cfzhou
"""

# libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize Layers
clas = Sequential()


# Convolution
clas.add(Conv2D(filters=8, 
                kernel_size=(3, 3),
                strides=1,
                padding="same",
                input_shape=(256, 256, 3),
                activation="relu",
                ))

# Pooling
clas.add(MaxPooling2D(pool_size=(2,2),
                      strides=2,
                      ))

# repeat convolution and pooling
conv = 5
for i in range(0,conv):
    # Convolution
    clas.add(Conv2D(filters=32*(2**(conv+1)), 
                kernel_size=(3, 3),
                # strides=1,
                # padding="same",
                activation="relu",
                ))

    # Pooling
    clas.add(MaxPooling2D(pool_size=(2,2),
                      strides=2,
                      ))

# Flattening
clas.add(Flatten())


# Full Connection
hidden=10

for i in range(0,hidden):
    clas.add(Dense(units=256,
               activation='relu',
               ))

# Output
clas.add(Dense(units=1,
               activation='sigmoid',
               kernel_initializer="uniform",
               ))

# compile
clas.compile(optimizer = 'adam', 
             loss='binary_crossentropy',
             metrics=['accuracy'])


# fitting & augmentation
from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

test = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

clas.fit_generator(
        train,
        steps_per_epoch=8000,
        epochs=50,
        validation_data=test,
        validation_steps=800)


# Save model
clas.save('LooCNN.h5')










