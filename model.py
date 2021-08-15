import csv 
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Flatten, Dense, Lambda, Activation, MaxPooling2D, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential, Model
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


lines = []
images = []
measurements = []  


with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    
    next(reader)   ## skip csv header
    
    for line in reader:
      #  steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
       #     correction = 0.2 # this is a parameter to tune
        #    steering_left = steering_center + correction
         #   steering_right = steering_center - correction

            # read in images from center, left and right cameras
          #  path = "../data/IMG/" # fill in the path to your training IMG directory
           # img_center = process_image(np.asarray(Image.open(path + row[0])))
            #img_left = process_image(np.asarray(Image.open(path + row[1])))
           # img_right = process_image(np.asarray(Image.open(path + row[2])))

            # add images and angles to data set
            #car_images.extend(img_center, img_left, img_right)
            #steering_angles.extend(steering_center, steering_left, steering_right)

        lines.append(line)
        
for line in lines:
    for i in range(3):
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = '../data/IMG/' + filename
        image = ndimage.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(images)
y_train = np.array(measurements)

# Neural Network Starts here
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2, 2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2, 2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2, 2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))




model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
model.save('model.h5')
