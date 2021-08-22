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


def process_image(image):
    # do some pre processing on the image
    #Contrast limited adaptive histogram equalization (CLAHE) is used for improve the visibility level of foggy image or video
    # TODO: 
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit = 5)
    final_image = clahe.apply(image_bw) + 30
    return final_image


lines = []
images = []
measurements = []  
number_of_epochs = 3
csv_file ='../data/driving_log.csv'

with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    
    next(reader)   ## skip csv header
    
    for line in reader:
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

#Flipping Images And Steering Measurements
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

    
with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            
            steering_center = float(row[3])
          
            # create adjusted steering measurements for the side camera images
            correction = 0.2 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            print(steering_center)
            # read in images from center, left and right cameras      
            path = "../data/" # fill in the path to your training IMG directory
            img_center = process_image(np.asarray(Image.open(path+row[0])))
            img_left = process_image(np.asarray(Image.open(path+row[1])))
            img_right = process_image(np.asarray(Image.open(path+row[2])))

            # add images and angles to data set
            car_images.extend(img_center, img_left, img_right)
            steering_angles.extend(steering_center, steering_left, steering_right)

X_train = np.array(images)
y_train = np.array(measurements)
  


# Neural Network Starts here
# The model I used is based on Nvidia's End to End Learning for Self-Driving Cars (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
# The Test Neural Network in Keras can be found below.

model = Sequential()

# set up lambda layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# crop out parts of the image where trees are(top) and hood of car(bottom) with cropping2D layer
model.add(Cropping2D(cropping=((70,25), (0,0))))

# starts with a convolutional and maxpooling layers
model.add(Convolution2D(24,5,5,subsample=(2, 2),activation="relu"))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

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
