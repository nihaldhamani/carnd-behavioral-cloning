import os
import csv

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt
import math

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

images = []
angles = []

correction = 0.2

for sample in samples[1:]:
    for i in range(3):
        source_path = sample[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        if i == 0:
            angles.append(float(sample[3]))
        elif i == 1:
            angles.append(float(sample[3]) + 0.2)
        else:
            angles.append(float(sample[3]) - 0.2)
    
aug_images, aug_angles = [], []
for image,angle in zip(images, angles):
    aug_images.append(image)
    aug_angles.append(angle)
    aug_images.append(cv2.flip(image,1))
    aug_angles.append(-1.0 * angle)
    

# compile and train the model using the generator function
X_train = np.array(aug_images)
y_train = np.array(aug_angles)

#ch, row, col = 3, 80, 320  # Trimmed image format

ch, row, col = 3, 160, 320 
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255 - 0.5,
        input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)
                  
model.save('model.h5')   

exit()

