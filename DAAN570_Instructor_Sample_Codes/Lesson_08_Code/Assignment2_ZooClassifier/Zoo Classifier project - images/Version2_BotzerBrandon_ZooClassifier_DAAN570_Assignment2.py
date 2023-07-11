# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 19:24:06 2023

@author: btb51
"""

#%%
# Imports

import os, shutil
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import image_dataset_from_directory


import numpy as np
import matplotlib.pyplot as plt

#%%
# Data Preprocessing:
    
'''
It seems that all of the images have different dimensions.
I will use the resize_with_pad to push everything to the largest m x n dim,
and then I will pool down from there with the goal of the padding 
pixels falling out in the CNN.
'''


image_h = 500
image_w = 500
batch_size = 16  #GPU Saturated and memory issues if I go to 32...

# location on disk of the image data
loc = 'C:/Users/btb51/Documents/GitHub/DeepLearning_DAAN570/DAAN570_Instructor_Sample_Codes/Lesson_08_Code/Assignment2_ZooClassifier/Zoo Classifier project - images/images'

#datasets will be a tuple of the train and validation data
train_ds, val_ds = image_dataset_from_directory(loc,
                              batch_size=batch_size,
                              image_size = (image_h,image_w),  # set as largest dims
                              shuffle = True,
                              seed = 570,
                              validation_split = 0.2,
                              subset = 'both')



#%%
class_names = train_ds.class_names
print(class_names)


#%%


plt.figure(figsize=(10, 10))
for images, labels in train_ds_pad.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(int(labels[i]))
    plt.axis("on")


#%%
# To manually iterate over the dataset to retrieve batches use the following:
    
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break



#%%

# Try to call numpy to get rid of the PrefetchDataset error on training

train_ds = np.array(train_ds)

val_ds = np.array(val_ds)

#%%
# Dataset performance and catche onto disk

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#%%

# Standardize the data down to [0, 1] range for the NN

normalization_layer = tf.keras.layers.Rescaling(1./255)

#%%

# Model building

num_classes = len(class_names)

net = Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(image_h, image_w, 3)),
    tf.keras.layers.Conv2D(16,3,padding='same', activation = 'relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation = 'relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation= 'relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(num_classes)
    ])


#%%


lr = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #Use sparce since these are integer encodings (not one-hot)  #does this need logits=True
metrics = ['accuracy']  #for some reason using tf.keras.metrics.Accuracy(), tf.keras.metrics.AUC() does not work... I don't know why...

# compile the model
net.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)



#%%
net.summary()

#%%

# Set the epochs and fit and take metrics
epochs=10

history = net.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)


#%%

def plot_acc_metric(history, title:str):
    

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(epochs)
    
    plt.figure(figsize=(8, 8))
    
    plt.suptitle(title)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training  Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training  Loss')
    plt.show()
    
#%%

plot_acc_metric(history, 'Small Generic CNN')





#%% Let try to take the image data and resize_with_pad in the begining





#%%
# Data Preprocessing:
    
'''
It seems that all of the images have different dimensions.
I will use the resize_with_pad to push everything to the largest m x n dim,
and then I will pool down from there with the goal of the padding 
pixels falling out in the CNN.
'''


image_h = 500
image_w = 500
batch_size = 25  #GPU Saturated and memory issues if I go to 32...

# location on disk of the image data
loc = 'C:/Users/btb51/Documents/GitHub/DeepLearning_DAAN570/DAAN570_Instructor_Sample_Codes/Lesson_08_Code/Assignment2_ZooClassifier/Zoo Classifier project - images/images'

#datasets will be a tuple of the train and validation data

# In an effort to try and reduce distortions from the resize
# crop_to_aspect ratio will return th elargest possible window
# of size image_size that matches the target aspect ratio.
train_ds_pad, val_ds_pad = image_dataset_from_directory(loc,
                              batch_size=batch_size,
                              image_size = (image_h,image_w),  # set as largest dims
                              shuffle = True,
                              seed = 570,
                              validation_split = 0.2,
                              subset = 'both',
                              crop_to_aspect_ratio = True)




#%%
AUTOTUNE = tf.data.AUTOTUNE

train_ds_pad = train_ds_pad.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds_pad = val_ds_pad.cache().prefetch(buffer_size=AUTOTUNE)




#%%

# should the validation_data be normalized?
val_ds_pad = val_ds_pad.nor


#%%

# Set the epochs and fit and take metrics
epochs=10

history_pad = net.fit(
    train_ds_pad,
    validation_data=val_ds_pad,
    epochs=epochs)

#%%

plot_acc_metric(history_pad, 'Generic CNN Using Crop_to_Aspect_ratio')

# Not really any noticable change here... CNN seems to be overfitting the data
# This most likely means that my network is too complex...

#%%

# I'll attempt some data augmentation
image_h = 500
image_w = 500
batch_size = 25  #GPU Saturated and memory issues if I go to 32...

# location on disk of the image data
loc = 'C:/Users/btb51/Documents/GitHub/DeepLearning_DAAN570/DAAN570_Instructor_Sample_Codes/Lesson_08_Code/Assignment2_ZooClassifier/Zoo Classifier project - images/images'

#datasets will be a tuple of the train and validation data

train_ds_aug = image_dataset_from_directory(loc,
                              batch_size=batch_size,
                              image_size = (image_h,image_w),  # set as largest dims
                              shuffle = True,
                              seed = 570,
                              validation_split = 0.2,
                              subset = 'training')

val_ds_aug = image_dataset_from_directory(loc,
                              batch_size=batch_size,
                              image_size = (256,256),  # set as largest dims
                              shuffle = True,
                              seed = 570,
                              validation_split = 0.2,
                              subset = 'validation')




#%%
# Run Data Augmentation to increase the robustness of the dataset

data_augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.5),
    tf.keras.layers.RandomCrop(256,256)
    ])



#Run the data_augment function on each image with its label
train_ds_aug = train_ds_aug.map(
    lambda image, label: (data_augment(image), label),
    num_parallel_calls=tf.data.AUTOTUNE)

# DO NOT AUGMENT YOUR VALIDATION DATA!



#%%

# MOdified Generic CNN for new augmented data size

# Model building

num_classes = len(class_names)

net_aug = Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(256, 256, 3)),
    tf.keras.layers.Conv2D(16,3,padding='same', activation = 'relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation = 'relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation= 'relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(num_classes)
    ])

#This new model with the smaller input shape is now ~ 8 million params
# the old generic CNN was ~ 31 million params

#%%


lr = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #Use sparce since these are integer encodings (not one-hot)  #does this need logits=True
metrics = ['accuracy']  #for some reason using tf.keras.metrics.Accuracy(), tf.keras.metrics.AUC() does not work... I don't know why...

# compile the model
net_aug.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)



#%%
net_aug.summary()






#%%
# Set the epochs and fit and take metrics
epochs=20

history_aug = net_aug.fit(
    train_ds_aug,
    validation_data=val_ds_aug,
    epochs=epochs)



#%%
plot_acc_metric(history_aug, 'Generic CNN with Aug Data')

# This model trained much slower and did not reach the same training acc
# However, while the validation acc is lower, it does seem to be
# slightly better at matching the training loss which would imply
# that we are doing a better job at not overfitting
# Try with 20 epochs 





#%%
# See what an augmented image looks like

plt.figure(figsize=(10, 10))
for images, labels in train_ds_pad.take(1):
    images = data_augment(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("on")

#%%
#Check for balanced dataset

cat = 0
dog =0
pan = 0

for _, labels in train_ds_pad:
    for each in labels:
        if each == 0:
           cat = cat +1
        elif each ==1:
            dog = dog +1
        elif each ==2:
            pan = pan +1
print("Cats, dogs, pandas")
print(cat, dog, pan)

#%%
#Check for balanced dataset; maybe this is giving me my problem

cat = 0
dog =0
pan = 0

for _, labels in val_ds_pa_ds_pad:
    for each in labels:
        if each == 0:
           cat = cat +1
        elif each ==1:
            dog = dog +1
        elif each ==2:
            pan = pan +1
print("Cats, dogs, pandas")
print(cat, dog, pan)

#datasets are balanced... this is not the problem

#%%        

# Then try with no cropping

# Run Data Augmentation to increase the robustness of the dataset

data_augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"),
    tf.keras.layers.RandomRotation((-0.25, 0.25))
    ])



#Run the data_augment function on each image with its label
train_ds_aug = train_ds_aug.map(
    lambda image, label: (data_augment(image), label),
    num_parallel_calls=tf.data.AUTOTUNE)

# DO NOT AUGMENT YOUR VALIDATION DATA!




#%%

# I'll attempt some data augmentation
image_h = 500
image_w = 500
batch_size = 25  #GPU Saturated and memory issues if I go to 32...

# location on disk of the image data
loc = 'C:/Users/btb51/Documents/GitHub/DeepLearning_DAAN570/DAAN570_Instructor_Sample_Codes/Lesson_08_Code/Assignment2_ZooClassifier/Zoo Classifier project - images/images'

#datasets will be a tuple of the train and validation data

train_ds_aug2 = image_dataset_from_directory(loc,
                              batch_size=batch_size,
                              image_size = (image_h,image_w),  # set as largest dims
                              shuffle = True,
                              seed = 570,
                              validation_split = 0.2,
                              subset = 'training')

val_ds_aug2 = image_dataset_from_directory(loc,
                              batch_size=batch_size,
                              image_size = (image_h, image_w),  # set as largest dims
                              shuffle = True,
                              seed = 570,
                              validation_split = 0.2,
                              subset = 'validation')


#%%

epochs=10

history = net.fit(
    train_ds_aug2,
    validation_data=val_ds_aug2,
    epochs=epochs)

#%%
plot_acc_metric(history, 'Generic CNN Data Aug, no crop')

# Same issue of overfitting with ~ 31 million params...








#%%

# Data for VGG, batch size is smaller due to GPU constraints
# I'll attempt some data augmentation
image_h = 500
image_w = 500
batch_size = 16  #GPU Saturated and memory issues if I go to 32...

# location on disk of the image data
loc = 'C:/Users/btb51/Documents/GitHub/DeepLearning_DAAN570/DAAN570_Instructor_Sample_Codes/Lesson_08_Code/Assignment2_ZooClassifier/Zoo Classifier project - images/images'

#datasets will be a tuple of the train and validation data

train_ds_aug3 = image_dataset_from_directory(loc,
                              batch_size=batch_size,
                              image_size = (image_h,image_w),  # set as largest dims
                              shuffle = True,
                              seed = 570,
                              validation_split = 0.2,
                              subset = 'training')

val_ds_aug3 = image_dataset_from_directory(loc,
                              batch_size=batch_size,
                              image_size = (image_h, image_w),  # set as largest dims
                              shuffle = True,
                              seed = 570,
                              validation_split = 0.2,
                              subset = 'validation')

#%%
#Run the data_augment function on each image with its label
train_ds_aug3 = train_ds_aug.map(
    lambda image, label: (data_augment(image), label),
    num_parallel_calls=tf.data.AUTOTUNE)

#%%






#%%


# Let's try VGG and see what happens... maybe the issue is my Generic CNN arch.

# CNN block (as VGG)

def cnn_block(num_conv, num_chan):
    
    block = Sequential()
    
    for _ in range(num_conv):
        
        block.add(tf.keras.layers.Conv2D(num_chan, kernel_size = 3,
                                         padding='same', activation='relu'))
        block.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        
    return block

# Convolutional Blocks Architecture as VGG 

conv_arch = ((1,64), (1,128), (2,256), (2,512), (2,512))

# Make the CNN as VGG

def vgg(conv_arch):
    
    cnn = tf.keras.models.Sequential()
    
    
    # If I wanted to preprocess the images in the Network
    # happens on the GPU (make sure your device GPU is configured)
    cnn.add(tf.keras.layers.Rescaling(1.0/255))  #rescale:
                                                    # with sequential this 
                                                    # should take any dim
                                                    # 500x500x3 x batch?
    
    for(num_conv, num_chan) in conv_arch:
        cnn.add(cnn_block(num_conv, num_chan))
        
    cnn.add(
        tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3)
            ]))
    
    # Think about adding the following:
        # Global Average 2D Pooling - tf.keras.layers.GlobalAvgPool2D()
        # 1x1 convolutions - 
    
    return cnn


#build the network!
net_vgg = vgg(conv_arch)
#%%

net_vgg.compile(optimizer=optimizer, loss = loss_fn, metrics=metrics)



#%%
net_vgg.summary()
#%%

history = net_vgg.fit(train_ds_aug3,
                      validation_data=val_ds_aug3,
                      epochs=10)

#%%
plot_acc_metric(history, 'VGG Run')

# It seems that the loss decreased slightly but not actual training was done...
# Maybe this is a learning rate issue?

#%%
import optuna

from keras.backend import clear_session

#%%

#Try to use optuna to get something going here in regards to learning rates


# Define the objective to run (essentially the code from above)
def objective_vgg(trial):
    
    clear_session()
    
    # load data
    
    image_h = 500
    image_w = 500
    batch_size = 16  #GPU Saturated and memory issues if I go to 32...

    # location on disk of the image data
    loc = 'C:/Users/btb51/Documents/GitHub/DeepLearning_DAAN570/DAAN570_Instructor_Sample_Codes/Lesson_08_Code/Assignment2_ZooClassifier/Zoo Classifier project - images/images'

    #datasets will be a tuple of the train and validation data

    train_ds_opt = image_dataset_from_directory(loc,
                                  batch_size=batch_size,
                                  image_size = (image_h,image_w),  # set as largest dims
                                  shuffle = True,
                                  seed = 570,
                                  validation_split = 0.2,
                                  subset = 'training')

    val_ds_opt = image_dataset_from_directory(loc,
                                  batch_size=batch_size,
                                  image_size = (image_h, image_w),  # set as largest dims
                                  shuffle = True,
                                  seed = 570,
                                  validation_split = 0.2,
                                  subset = 'validation')
    
    # Trial different values for lr    
    lr_trial = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    
    # standard items to compile
    optimizer_trial = tf.keras.optimizers.Adam(learning_rate=lr_trial)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #Use sparce since these are integer encodings (not one-hot)  #does this need logits=True
    metrics = ['accuracy'] 
    
    
    net_vgg.compile(optimizer=optimizer, loss = loss_fn, metrics=metrics)
    
    net_vgg.fit(train_ds_opt,
              validation_data=val_ds_opt,
              epochs=5)
    
    score = net_vgg.evaluate(val_ds_opt, verbose = 0)
    return score[1]


#%%

# Create the 'study' object
study = optuna.create_study(direction="maximize")

# Run the optimize using the 'objective' defined above
study.optimize(objective_vgg, n_trials=5, timeout = 600)

print("Number of finished Trials: {}".format(len(study.trials)))

print("Best Trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
# This didn't change much of anything...
'''
Best Trial:
  Value: 0.3466666638851166
  Params: 
    learning_rate: 0.0018305783879910714
'''



#%%

# Go back to using the overfitting Generic CNN but add dropout
num_classes = len(class_names)

def net_drop(dropout):

    net_drop = Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(image_h, image_w, 3)),
        tf.keras.layers.Conv2D(16,3,padding='same', activation = 'relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation = 'relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation= 'relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(num_classes)
        ])

    return net_drop

#%%
def objective_cnn_dropper(trial):
    
    clear_session()
    
    # load data
    
    image_h = 500
    image_w = 500
    batch_size = 16  #GPU Saturated and memory issues if I go to 32...

    # location on disk of the image data
    loc = 'C:/Users/btb51/Documents/GitHub/DeepLearning_DAAN570/DAAN570_Instructor_Sample_Codes/Lesson_08_Code/Assignment2_ZooClassifier/Zoo Classifier project - images/images'

    #datasets will be a tuple of the train and validation data

    train_ds_opt = image_dataset_from_directory(loc,
                                  batch_size=batch_size,
                                  image_size = (image_h,image_w),  # set as largest dims
                                  shuffle = True,
                                  seed = 570,
                                  validation_split = 0.2,
                                  subset = 'training')

    val_ds_opt = image_dataset_from_directory(loc,
                                  batch_size=batch_size,
                                  image_size = (image_h, image_w),  # set as largest dims
                                  shuffle = True,
                                  seed = 570,
                                  validation_split = 0.2,
                                  subset = 'validation')
    
    # Trial different values for lr    
    lr_trial = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    
    dropout_trial = trial.suggest_float("dropout rate", 0.05, 0.5, log = True)
    
    # standard items to compile
    optimizer_trial = tf.keras.optimizers.Adam(learning_rate=lr_trial)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #Use sparce since these are integer encodings (not one-hot)  #does this need logits=True
    metrics = ['accuracy'] 
    
    
    net_dr= net_drop(dropout_trial)
    
    net_dr.compile(optimizer=optimizer, loss = loss_fn, metrics=metrics)
    
    net_dr.fit(train_ds_opt,
              validation_data=val_ds_opt,
              epochs=5)
    
    score = net_vgg.evaluate(val_ds_opt, verbose = 0)
    return score[1]
    
#%%

# Create the 'study' object
study = optuna.create_study(direction="maximize")

# Run the optimize using the 'objective' defined above
study.optimize(objective_cnn_dropper, n_trials=5, timeout = 6000)
#%%
print("Number of finished Trials: {}".format(len(study.trials)))

print("Best Trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

#%%

'''
The best I found was before the memory crashed:
    Number of finished Trials: 5
    Best Trial:
      Value: 0.33666667342185974
      Params: 
        learning_rate: 0.07056230658689876
        dropout rate: 0.3956136450158745
'''


#%%


net_best = net_drop(0.39)


# standard items to compile
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #Use sparce since these are integer encodings (not one-hot)  #does this need logits=True
metrics = ['accuracy'] 

net_best.compile(optimizer=optimizer, loss = loss_fn, metrics=metrics)

net_best.summary()

#%%

history_best = net_best.fit(train_ds, validation_data=val_ds, epochs = 10)

#%%

plot_acc_metric(history_best, 'Generic CNN with Best Hyperparms')







#%%

from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import SeparableConv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class CancerNet:
    @staticmethod
    def build(width, height, depth, classes):
        model=Sequential()
        shape=(height, width, depth)
        channelDim=-1
        
        if K.image_data_format() == 'channels_first':
            shape=(depth,height,width)
            channelDim=1
            
        model.add(SeparableConv2D(32, (3,3), padding='same', input_shape=shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.23))
        
        model.add(SeparableConv2D(64,(3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(SeparableConv2D(64,(3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        model.add(SeparableConv2D(128, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(SeparableConv2D(128, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(SeparableConv2D (128, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        return model













