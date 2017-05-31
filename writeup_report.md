# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/history.png "History"
[image3]: ./examples/center.jpg "Center Image"
[image4]: ./examples/zigzag0.jpg "Zigzag Image"
[image5]: ./examples/zigzag1.jpg "Zigzag Image"
[image6]: ./examples/zigzag2.jpg "Zigzag Image"
[image7]: ./examples/flip.jpg "Flipped Image"
[image8]: ./examples/backward.jpg "Backward Image"

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

I have written for two models, LeNet & NVIDIA. And I've confirmed the NVIDIA model works a lot better in practice.
LeNet model consists of a CNN with 5x5 filter sizes and depths of 6. (model.py lines 96-102).
NVIDIA model consists of a CNN with 5x5, 3x3 filter sizes and depths between 24 and 64 (model.py lines 106-114) 

Both model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 91). The image is cropped using Keras Cropping2D layer(code line 92).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 76). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 118).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving(track1_normal1), zig-zag driving(track1_normal2), and backward driving(track1_backward).

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use well-known model.

My first step was to use LeNet I thought this model might be appropriate because it is relatively simple to use.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on both training set and a validation set. This implied that the model was trained well.

The final step was to run the simulator to see how well the car was driving around track one. The vehicle fell off the track quite often. To improve the driving behavior, I've recorded more training data and augmented by flipping images and angles.

At the end of the process, the vehicle is able to drive autonomously around the track, but there still remained some spots, where the vehicle fell off.

#### 2. Final Model Architecture

The final model architecture (model.py lines 106-114) consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle driving zigzag on the road so that the vehicle would learn to drive recovering:

![alt text][image4]
![alt text][image5]
![alt text][image6]

Then I recorded backward driving to get more data points: 
![alt text][image8]

To augment the data set, I also flipped images and angles thinking that this would generalize train set data. For example, here is an image that has then been flipped:

![alt text][image7]


After the collection process, I had 10924 number of data points. I then preprocessed this data by adding left and right images and flipped. The final data points are 65544.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 - 12 as evidenced by Keras EarlyStopping callbacks. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here is the visualization of history generated by fit_generator:
![alt text][image2]
