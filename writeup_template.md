# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


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

The model I used was Nvidia self-driving model. Before feed the data into the model, I used a Lambda layer to normalize the input images and a Cropping2D layer to crop the useful part of the images. Then, the model was followed by 5 convolutional layers, 1 flatten layer, and 4 fully connected layers.

#### 2. Attempts to reduce overfitting in the model

Because the epoches number I used was not that large, which is 10. By testing couples different approaches(with or without dropout, different models, different epoches, different driving style of training data), I decide not to use drop out layer based on the best experimental result.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 71).

#### 4. Appropriate training data

I used to simulator to collect about 5 loops of training data. 3 loops od normal driving with the vehicle stay on the center of the road, 1 loop of recovery activaties(aggressive turns), 1 loop of staying close to the lanes.

I tried couples different driving strategies. The first one was collecting as much data as possible. In this case, I drove about more than 10 loops, with many kind of abnormal actions in order to train the model learn how to recover from various situation. But the result was pretty bad. Because the model learned my unusual actions and running out from the track at the very begining. The most significant improvement I made was using side camera images. In the case that I didn't use the side camera, it doesn't matter how many data I collected or how great the model was, the result was always bad, because when the vehicle was close to the side of the road and it was trying to turn back to the center, but the angle was too small to recover. By adding the side camera images, there were much more data with large turning angle than before and the model could handle the turning pretty well.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I tried Lenet before used the Nvidia net, but the result wasn't that great. Then I used the Nvidia net directly, because I guessed the Nvidia net might be the best to handle this kind of data, otherwise we won't learn it from the lecture. I didn't change the main architecture, but only add two dropout layers for testing purpose. But the final model I used was exactly the same as Nvidia net.

#### 2. Final Model Architecture
`
* model = Sequential()
* model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
* model.add(Cropping2D(cropping=((50,20),(0,0))))
* model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
* model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
* model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
* model.add(Convolution2D(64,3,3,activation="relu"))
* model.add(Convolution2D(64,3,3,activation="relu"))
* model.add(Flatten())
* model.add(Dense(100))
`

#### 3. Creation of the Training Set & Training Process

* My first step was only use one loop of center driving to train the Nvidia net, but the vehicle don't know how to make a turn.
* Then I added about 10 loops of normal and abnormal driving data, but the vehicle run out of the road in the very begining.
* Looks like a small amount of bad data will damage the model even though there are a lot of more good data. So I create a new folder to collect only good behavior and a little bit of recover action. The result was really well.
