#!/usr/bin/env python
# coding: utf-8

import csv
from time import time
from math import ceil
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Cropping2D, MaxPooling2D, Lambda, BatchNormalization, Dropout, Activation
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


STEERING_EPSILON = 0.15

print("Loading data.")
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    cnt = 0
    for line in reader:
        cnt +=1
        if cnt == 1:
            continue
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=20):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center,left,right,steering,throttle,brake,speed = [token.strip() for token in batch_sample]
                try:
                    steering = float(steering)
                except ValueError:
                    continue
                    print("error Value: ", steering)
                left_side_steering = steering + STEERING_EPSILON
                right_side_steering = steering - STEERING_EPSILON
                center_image = mpimg.imread('data/'+center)
                left_image = mpimg.imread('data/'+left)
                right_image = mpimg.imread('data/'+right)

                images.append(center_image)
                images.append(np.fliplr(center_image))
                images.append(left_image)
                images.append(np.fliplr(left_image))
                images.append(right_image)
                images.append(np.fliplr(right_image))

                angles.append(steering)
                angles.append(steering*-1.0)
                angles.append(left_side_steering)
                angles.append(left_side_steering*-1.0)
                angles.append(right_side_steering)
                angles.append(right_side_steering*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)



EPOCHS = 50
BATCH_SIZE = 24
NUM_AUG = 6 # Gives how many images per datapoint are added. This constant defines the degree to which augmentation is done.

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)


model = Sequential()
# Preprocess
# model.add(Lambda(lambda x: tf.image.rgb_to_yuv(x), ))
model.add(Cropping2D(cropping=((70, 20), (0, 0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
# Conv layer 1
model.add(Conv2D(24,(5,5), padding = "same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Conv layer 2
model.add(Conv2D(36,(5,5), padding = "same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Conv layer 3
model.add(Conv2D(48,(5,5), padding = "same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Conv layer 4
model.add(Conv2D(64,(3,3), padding = "same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
# Conv layer 5
model.add(Conv2D(64,(3,3), padding = "same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
# Flatten the output
model.add(Flatten())
# Dense layer 1
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
# Dense layer 2
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
# Dense layer 3
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')


for i in range(0, EPOCHS):
    print("Starting batch: ", i)
    model.fit_generator(train_generator, 
        steps_per_epoch=ceil((len(train_samples)*NUM_AUG)/BATCH_SIZE),
        validation_data=validation_generator, 
        validation_steps=ceil((len(validation_samples)*NUM_AUG)/BATCH_SIZE), 
        shuffle=True, 
        epochs=1,
        verbose=1)
    model.save('model_'+str(i)+'_'+str(time())[:-8]+'.h5')

# This training process gave the optimal trained model - model_6_1569439496.h5
