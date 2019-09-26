#!/usr/bin/env python
# coding: utf-8

import csv
from time import time
from math import ceil
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, Cropping2D, MaxPooling2D, Lambda, BatchNormalization, Dropout, Activation
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


STEERING_EPSILON = 0.15

print("Loading data.")
samples = []
with open('./collected_data/driving_log.csv') as csvfile:
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
                pos = center.find('IMG')
                center_image = mpimg.imread('collected_data/'+center[pos:])

                images.append(center_image)
                angles.append(steering)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)



EPOCHS = 50
BATCH_SIZE = 24
NUM_AUG = 6 # Gives how many images per datapoint are added. This constant defines the degree to which augmentation is done.

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)


model = load_model('model_6_1569439496.h5')

for i in range(0, EPOCHS):
    print("Starting batch: ", i)
    model.fit_generator(train_generator, 
        steps_per_epoch=ceil((len(train_samples)*NUM_AUG)/BATCH_SIZE),
        validation_data=validation_generator, 
        validation_steps=ceil((len(validation_samples)*NUM_AUG)/BATCH_SIZE), 
        shuffle=True, 
        epochs=1,
        verbose=1)
    model.save('model_tuned_'+str(i)+'_'+str(time())[:-8]+'.h5')

# This training process gave the optimal trained model - model_tuned_14_156945148.h5
# This is renamed as model.h5

'''
Here's the log for model tuning-

Starting batch:  0 
Epoch 1/1
18/18 [==============================] - 13s 706ms/step - loss: 0.0360 - val_loss: 0.0475
Starting batch:  1 
Epoch 1/1
18/18 [==============================] - 11s 626ms/step - loss: 0.0340 - val_loss: 0.0437
Starting batch:  2 
Epoch 1/1
18/18 [==============================] - 12s 671ms/step - loss: 0.0327 - val_loss: 0.0429
Starting batch:  3 
Epoch 1/1
18/18 [==============================] - 12s 662ms/step - loss: 0.0313 - val_loss: 0.0427
Starting batch:  4 
Epoch 1/1
18/18 [==============================] - 12s 676ms/step - loss: 0.0310 - val_loss: 0.0424
Starting batch:  5 
Epoch 1/1
18/18 [==============================] - 12s 647ms/step - loss: 0.0292 - val_loss: 0.0422
Starting batch:  6 
Epoch 1/1
18/18 [==============================] - 12s 659ms/step - loss: 0.0284 - val_loss: 0.0424
Starting batch:  7 
Epoch 1/1
18/18 [==============================] - 12s 655ms/step - loss: 0.0289 - val_loss: 0.0423
Starting batch:  8 
Epoch 1/1
18/18 [==============================] - 12s 667ms/step - loss: 0.0273 - val_loss: 0.0428
Starting batch:  9 
Epoch 1/1
18/18 [==============================] - 12s 671ms/step - loss: 0.0267 - val_loss: 0.0422
Starting batch:  10 
Epoch 1/1
18/18 [==============================] - 12s 642ms/step - loss: 0.0274 - val_loss: 0.0419
Starting batch:  11 
Epoch 1/1
18/18 [==============================] - 12s 691ms/step - loss: 0.0259 - val_loss: 0.0435
Starting batch:  12 
Epoch 1/1
18/18 [==============================] - 12s 662ms/step - loss: 0.0254 - val_loss: 0.0416
Starting batch:  13 
Epoch 1/1
18/18 [==============================] - 12s 683ms/step - loss: 0.0261 - val_loss: 0.0409
Starting batch:  14 
Epoch 1/1
18/18 [==============================] - 12s 679ms/step - loss: 0.0229 - val_loss: 0.0408
Starting batch:  15 
Epoch 1/1
18/18 [==============================] - 12s 667ms/step - loss: 0.0235 - val_loss: 0.0430
Starting batch:  16 
Epoch 1/1
18/18 [==============================] - 11s 631ms/step - loss: 0.0228 - val_loss: 0.0422
Starting batch:  17 
Epoch 1/1
18/18 [==============================] - 11s 587ms/step - loss: 0.0211 - val_loss: 0.0429
Starting batch:  18 
Epoch 1/1
18/18 [==============================] - 15s 855ms/step - loss: 0.0213 - val_loss: 0.0420
Starting batch:  19 
Epoch 1/1
18/18 [==============================] - 16s 907ms/step - loss: 0.0186 - val_loss: 0.0446
Starting batch:  20 
Epoch 1/1
18/18 [==============================] - 16s 916ms/step - loss: 0.0189 - val_loss: 0.0428
Starting batch:  21 
Epoch 1/1
18/18 [==============================] - 16s 911ms/step - loss: 0.0186 - val_loss: 0.0422
Starting batch:  22 
Epoch 1/1
18/18 [==============================] - 13s 706ms/step - loss: 0.0177 - val_loss: 0.0446
Starting batch:  23 
Epoch 1/1
18/18 [==============================] - 11s 607ms/step - loss: 0.0159 - val_loss: 0.0432
Starting batch:  24 
Epoch 1/1
18/18 [==============================] - 12s 639ms/step - loss: 0.0155 - val_loss: 0.0471
'''
