import csv 
import os
import cv2
import numpy as np
# Setting up Tensorflow
import tensorflow as tf
# Setting up Keras
from keras.layers import Flatten, Dense, Lambda, Activation, MaxPooling2D, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential, Model
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import PurePosixPath

lines = []
images = []
measurements = []  
number_of_epochs = 5
csv_file ='../data/driving_log.csv'
steering_angles = []

def process_image(image):
    # do some pre processing on the image
    #Contrast limited adaptive histogram equalization (CLAHE) is used for improve the visibility level of foggy image or video
    # TODO: 
    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit = 5)
    final_image = clahe.apply(bw_image) + 30
    return final_image

with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    
    next(reader)   ## skip csv header
    
    for line in reader:
        lines.append(line)

#Reading in image data from IMG folder and measurement
for line in lines:
    for i in range(3):
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = '../data/IMG/' + filename
        image = ndimage.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

#Flipping Images And Steering Measurements this augments the training size by 3
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


# My Neural Network that using transfer learning from Nvidia's "End to End Learning for Self-Driving Cars" (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) 
# can be found below
model = Sequential()

# set up lambda layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# crop out parts of the image where trees are(top) and hood of car(bottom) with cropping2D layer
model.add(Cropping2D(cropping=((70,25), (0,0))))

# starts with a convolutional and maxpooling layers
model.add(Convolution2D(24,5,5,subsample=(2, 2),activation="relu"))
#MaxPooling was added to the network to fix issue of going off track after the bridge
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

# then a convolutional and maxpooling layer
model.add(Convolution2D(36,5,5,subsample=(2, 2),activation="relu"))


# then a convolutional
model.add(Convolution2D(48,5,5,subsample=(2, 2),activation="relu"))

# then another convolutional 
model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Flatten())

# Next, four fully connected layers
model.add(Dense(100))
#model.add(Activation(relu))
model.add(Dense(50))
#model.add(Activation(relu))
model.add(Dense(10))
#model.add(Activation(relu))
model.add(Dense(1))
#model.add(Activation(relu))

model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=number_of_epochs)
model.save('model.h5')

