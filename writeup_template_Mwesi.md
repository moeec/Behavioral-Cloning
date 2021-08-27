# **Behavioral Cloning** 

Introduction:

In this project, I will use what I've learned about deep neural networks and convolutional neural networks to clone driving behavior. I will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

I used a simulator where you can steer a car around a track for data collection. I will collect both image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


**Use the simulator to collect data of good driving behavior**

I run the Udacity simluator to record good driving data used to train the model. 
When driving I first went around the track 5 times in one recording then 4 times. Out of these 9 times I made sure I was staying within the road and in some instances I moved to the left and right side of the road then recentered to the center to demonstrate recovery to the neural network.
This process was then repeated for the left camera then the right camera teaching the neural network different viewpoints.



The dataset consists of 22134 images:
- 7377 Center Camera Image, 
- 7370 left Camera Image 
- 7387 Right Camera Image angle

The training track contains a lot of shallow turns and straight road segments. Hence, the majority of the recorded steering angles are zeros. Therefore, preprocessing images and respective steering angles are necessary in order to generalize the training model for unseen tracks such as our validation track.
[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image


**Build, a convolution neural network in Keras that predicts steering angles from images**
**Neural Network Archectiture**

| Layer(type)       | Output Shape        |# Of Parameters
| ------------------| --------------------|----------
| Lambda            | (160, 320, 3)       | 0
| Cropping2D        | (65, 320, 3)        | 0
| Convolution2D     | (33, 160, 24)       | 1824
| Activation        | (33, 160, 24)       | 0
| Max Pooling2D     | (32, 159, 24)       | 0
| Convolution2D     | (16, 80, 36)        | 21636
| Activation        | (16, 80, 36)        | 0
| Max Pooling2D     | (15, 79, 36)        | 0
| Convolution2D     | (8, 40, 48)         | 43248
| Activation        | (8, 40, 48)         | 0
| Max Pooling2D     | (7, 39, 48)         | 0
| Convolution2D     | (7, 39, 64)         | 27712
| Activation        | (7, 39, 64)         | 0
| Max Pooling2D     | (6, 38, 64)         | 0
| Convolution2D     | (6, 38, 64)         | 36928
| Activation        | (6, 38, 64)         | 0
| Max Pooling2D     | (5, 37, 64)         | 0
| Flatten           | (11840)             | 0
| Dense             | (1164)              | 13782924
| Activation        | (1164)              | 0
| Dense             | (100)               | 116500
| Activation        | (100)               | 0
| Dense             | (50)                | 5050
| Activation        | (50)                | 0  
| Dense             | (10)                | 510
| Activation        | (10)                | 0 
| Dense             | (1)                 | 11
| ------------------| --------------------|----------
|Total params: 14,036,343
|Trainable params: 14,036,343
|Non-trainable params: 0
| ---------------------------------------------------
|Train on 19286 samples, validate on 4822 samples


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
