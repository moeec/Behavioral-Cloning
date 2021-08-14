import csv 
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential


lines = []
images = []
measurements = []  


with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape)

# Neural Network Starts here
model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)
model.save('model.h5')