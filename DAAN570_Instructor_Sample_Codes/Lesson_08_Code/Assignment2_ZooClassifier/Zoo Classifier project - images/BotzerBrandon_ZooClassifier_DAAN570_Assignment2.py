# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 18:02:48 2023

@author: Brandon Botzer - btb5103

Building a Zoo classifier
Build a classifier to classify images onto one of three classes/labels. 
The dataset  Download datasetconsists of three folders (cats, dogs and panda), 
containing images in different dimensions.  Chose one of the following options 
to build your classifier.

Option 1: Build a model based on a convolutional neural network. 
For a classification problem, you usually choose softmax as output activation 
function and the categorical_crossentropy as a loss function.


Option 2: Build a model from a pre-trained model using transfer learning 
( VGG 19 model for example). Adapt your model to the image size 
(for example 150x150x3) and outputs 
(keras API has already a list of built-in pretrained deep networks).


Notes:

As in all deep neural networks, it is difficult to find out the 
optimal CNN model. That's why we usually build one or more neural network 
models with different architectures  
(i.e., different number of layers, different activation functions 
 and/or different number of neurons per layer) and  different hyperparameter 
values (i.e., learning rates, number of epochs, batch size, 
        optimizer (Adam, RMSProp, SGD) ). 
For this assignment, start by building a base line model and fit it to 
your training and testing datasets. Based on the evaluation of your 
loss/accuracy plots, you may decide to improve the model accuracy by varying 
the neural network architecture and/or hyper-parameter values 
(please refer to the lessons and their case-studies in Keras for examples).

Regardless your choice of option 1 or option 2, try to improve your the 
accuracy of model. Another important point is the data exploration. 
Before training your model, you should get insights from your dataset 
(number of images, view show couples of images, and specify the image 
 size to use in the training) . This is an indispensable and systematic 
task in any data analytics or machine learning projects.

 

Upload two files: One file is the Jupyter Notebook with your answers, 
ready to be executed whereas the second is its print out in PDF format.
"""

#%%
# Imports

import os, shutil
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import image_dataset_from_directory

import numpy as np
import matplotlib.pyplot as plt



#%%
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#%%
# CNN block (as VGG)

def cnn_block(num_conv, num_chan):
    
    block = Sequential()
    
    for _ in range(num_conv):
        
        block.add(tf.keras.layers.Conv2D(num_chan, kernel_size = 3,
                                         padding='same', activation='relu'))
        block.add(tf.kerasl.layers.MaxPool2D(pool_size=2, strides=2))
        
    return block
    
    
#%%
    
# Convolutional Blocks Architecture as VGG 

conv_arch = ((1,64), (1,128), (2,256), (2,512), (2,512))


#%%

# Make the CNN as VGG

def vgg(conv_arch):
    
    cnn = tf.keras.models.Sequential()
    
    for(num_conv, num_chan) in conv_arch:
        cnn.add(cnn_block(num_conv, num_chan))
        
    cnn.add(
        tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3, activation='softmax')
            ]))
    
    # Think about adding the following:
        # Global Average 2D Pooling - tf.keras.layers.GlobalAvgPool2D()
        # 1x1 convolutions - 
    
    return cnn

#%%

# Data Preprocessing:
    
'''
It seems that all of the images have different dimensions.
I will use the resize_with_pad to push everything to the largest m x n dim,
and then I will pool down from there with the goal of the padding 
pixels falling out in the CNN.
'''


image_size = (500,500)
batch_size = 150  #This makes 16 nice batches for the 2400 training data

# location on disk of the image data
loc = 'C:/Users/btb51/Documents/GitHub/DeepLearning_DAAN570/DAAN570_Instructor_Sample_Codes/Lesson_08_Code/Assignment2_ZooClassifier/Zoo Classifier project - images/images'

#datasets will be a tuple of the train and validation data
train_data, val_data = image_dataset_from_directory(loc,
                              labels='inferred',
                              label_mode='int',
                              color_mode = 'rgb',
                              batch_size=batch_size,
                              image_size = image_size,  # set as largest dims
                              shuffle = True,
                              seed = 570,
                              validation_split = 0.2,
                              subset = 'both')

    
#%%

# Data Visualization for Cats = 0, Dogs = 1, Pandas = 2

plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
    
    

#%%

    
    
    
    
    
    
    










#%%






    