# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./network_model.png "Model Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `model_tuned.py` that contains the code to fine-tune the model on more collected data
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing the final trained+tuned convolution neural network
* `model_6_1569439496.h5` that is the intermediate trained model 
* `writeup_report.md` summarizing the results
* `collected_data` and `data` (sample data) used for tuning and training the model respectively.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` and `model_tuned.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network similar to the NVIDIA's deep-learning model for training the car in this paper: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf.
The network contains 5 convolutional layers, followed by 3 hidden layers and a single unit output layer.

The model includes RELU layers to introduce nonlinearity (model.py), and the data is normalized in the model using a Keras lambda layer (model.py). 

#### 2. Attempts to reduce overfitting in the model

The model contains batch-normalization and dropout layers in order to reduce overfitting. 

The model was trained and validated on two data sets to ensure that the model was not overfitting. The tuned model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the default data set with augmentation of left and right cameras and horzontal flipping to first train the model thoroughly. 
But I saw that the car would leave the road on very sharp turns. So I collected extra data to model recovery behaviour and used that to tune the existing model.

Both times during training, I kept saving the intermittent model, and picked the model with the least validation losses for the next task. Therefore,
1. Using the model from the first training exercise with the lowest validation loss, I tuned it using the collected data
2. Using the model tuned with the collected data which had the least validation loss, I tested the car in autonomous mode in the simulator.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

A) Model
I read the paper linked to the NVIDIA's neural-network model, and used it as the default model to approach the problem. I simply added batch-normalization and drop-out wherever necessary to reduce overfitting and make the training faster.

B) Data
I wanted to check the baseline performance of the model, so I took the sample driving data, and as shown in the tutorials, added the left and right camera frames as well with correction to steering angles.
Also, I added the mirror images of the dataset, as the track is mostly turning towards left.

The overall performance of the model was satisfactory, except a few edge-cases where the car left the track. I though adding some data for recovery driving would be useful, so colledted the data in training mode and used the new dataset to tune the existing model.

C) Preprocessing
I normalized the images and cropped the unnecessary parts from the top and bottom areas as suggested in the tutorials. I tried to convert the RGB to YUV iamge format as well as suggested by the paper, but the function to do that was not available in the tensorflow version provided with the workspace.

D) Simulator
I checked the performance of the model both times on the simulator to see the results.

#### 2. Final Model Architecture

The final model architecture is the same as mentioned above:

1st layer - Conv 5x5 kernel and 24 output feature maps with batch-norm and (2,2) maxpooling

2nd layer - Conv 5x5 kernel and 36 output feature maps with batch-norm and (2,2) maxpooling

3rd layer - Conv 5x5 kernel and 48 output feature maps with batch-norm and (2,2) maxpooling

4th layer - Conv 3x3 kernel and 64 output feature maps with batch-norm 

5th layer - Conv 3x3 kernel and 64 output feature maps with batch-norm

6th layer - Linear layer with 100 relu neurons and dropout at 0.3

7th layer - Linear layer with 50 relu neurons and dropout at 0.3

8th layer - Linear layer with 10 relu neurons

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

As already mentioned, I mainly used the sample data to train the model to drive around the track within the lanes. I augmented this dataset by adding mirror images for each datapoint.

For edge cases, I recorded 2 laps where I kept recovering from turns towards the center of the road, and used this to tune the model. I did not flip these data points as it didn't seem necessary after testing.


For both the cases, I used a generator to feed the data into the model. There generator always returns a shuffled batch from the total samples to maintain randomness in each batch during training.

The first training procedure on sample data started overfitting after 5-6 epochs. I suspended training and chose the saved model version with the least validation loss.
Next, I used this trained model and again trained it using the new dataset. This time the model beyond 15 epochs. Again I suspended the training and then recorded the video based on the driving output generated through the predictions of this tuned model.