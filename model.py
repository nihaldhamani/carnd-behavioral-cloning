import os
import csv

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
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

for sample in samples[1:]:
    source_path = sample[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    angles.append(float(sample[3]))
    

# compile and train the model using the generator function
X_train = np.array(images)
y_train = np.array(angles)

#ch, row, col = 3, 80, 320  # Trimmed image format
ch, row, col = 3, 160, 320 
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255 - 0.5,
        input_shape=(row, col, ch)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
                  
model.save('model.h5')               
                
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

