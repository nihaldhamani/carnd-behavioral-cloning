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

[image1]: ./pics/pic1.jpg "Center Lane Image"
[image2]: ./pics/pic_left.jpg "Left Lane Image"
[image3]: ./pics/pic_right "Right Lange Image"

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

My model consists of a convolution neural network with 5 convolution layers sizes and followed by 3 fully connected layers (model.py lines 60-74) 

The model crops the incoming image to reduce noise from background and the data is normalized in the model using a Keras lambda layer (model.py 62-64). 

#### 2. Attempts to reduce overfitting in the model 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The number of epochs was strategically picked in order to avoid overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center camera, left camera, and right camera images in order to train a more robust model. Additionally, I flipped each image with cv2.flip() in order to provide even more training data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start of simple and iteratively increase complexity.

My first step was to use a single layer to make sure all my data preprocessing was working correctly. Slowly, I ended up implementing the LeNet architecture. Finally, I implemented the architecture described in NVIDIA's self driving car blog: [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Then I ran the simulation.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track so I made sure to have multiple driving runs that would successfully pass through those spots.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I also included images from the left and right camera:
![alt text][image2]
![alt text][image3]


To augment the data sat, I also flipped images and angles thinking that this would provide more data points and fix the class imbalance problem.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4. I used an adam optimizer so that manually training the learning rate wasn't necessary.
