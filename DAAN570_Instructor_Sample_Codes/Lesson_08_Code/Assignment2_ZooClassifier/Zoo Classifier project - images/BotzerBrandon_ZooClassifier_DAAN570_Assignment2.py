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
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import image_dataset_from_directory

import numpy as np
import matplotlib.pyplot as plt



#%%
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Is your GPU configured?

print(tf.config.list_physical_devices('GPU'))

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')

#%%
# CNN block (as VGG)

def cnn_block(num_conv, num_chan):
    
    block = Sequential()
    
    for _ in range(num_conv):
        
        block.add(tf.keras.layers.Conv2D(num_chan, kernel_size = 3,
                                         padding='same', activation='relu'))
        block.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        
    return block
    
    
#%%
    
# Convolutional Blocks Architecture as VGG 

conv_arch = ((1,64), (1,128), (2,256), (2,512), (2,512))


#%%

# Make the CNN as VGG

def vgg(conv_arch):
    
    cnn = tf.keras.models.Sequential()
    
    
    # If I wanted to preprocess the images in the Network
    # happens on the GPU (make sure your device GPU is configured)
    #cnn.add(tf.keras.layers.Rescaling(1.0/255))  #rescale:
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

# Can I make a plot of the model?  It is a sequential set up here

#plot_model(net)


# Test out the VGG to make sure it works...

X = tf.random.uniform((1,500,500,3))

for block in net.layers:
    X = block(X)
    print(block.__class__.__name__, 'output shape:\t', X.shape)
    
# This works but my own data does not... I bet it is the batching

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
batch_size = 16  #This makes 16 nice batches for the 2400 training data

# location on disk of the image data
loc = 'C:/Users/btb51/Documents/GitHub/DeepLearning_DAAN570/DAAN570_Instructor_Sample_Codes/Lesson_08_Code/Assignment2_ZooClassifier/Zoo Classifier project - images/images'

#datasets will be a tuple of the train and validation data
train_data, val_data = image_dataset_from_directory(loc,
                              labels='inferred',
                              label_mode='int',
                              color_mode = 'rgb',
                              batch_size=batch_size,
                              image_size = (image_h, image_w),  # set as largest dims
                              shuffle = True,
                              seed = 570,
                              validation_split = 0.2,
                              subset = 'both')

    
#%%

# Data Visualization for Cats = 0, Dogs = 1, Pandas = 2

plt.figure(figsize=(10, 10))

# From the train_data, take(1) image and its label
# plot these in the 3x3 grid
for images, labels in train_data.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
           

#%%

# Run Data Augmentation to increase the size of the dataset

data_augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.5)
    ])

# Think about putting in a random zoom later - tf.keras.layers.RandomZoom(0.1)


#%%



#%%
    
# Plot the image data with augmentation

plt.figure(figsize=(10,10))

for images, labels in train_data.take(1):
    for i in range(9):
        augmented_images = data_augment(images)
        ax = plt.subplot(3,3, i+1)
        plt.imshow(augmented_images[i].numpy().astype("uint8"))
        plt.title((int(labels[i])))
        plt.axis("off")  
     


#%%

# Proprocess the data asynchronously

# Happens on the CPU (you know this will work)
#aug_train_data = train_data.map(
#    lambda x, y: (data_augment(x, training=True), y))


#Run the data_augment function on each image with its label
train_data = train_data.map(
    lambda image, label: (data_augment(image), label),
    num_parallel_calls=tf.data.AUTOTUNE)

# DO NOT AUGMENT YOUR VALIDATION DATA!

#%%

# Prefetch samples in GPU memory for GPU utilization
# Make sure your GPU is config'd
'''
train_data = train_data.prefetch(tf.data.AUTOTUNE)
val_data = val_data.prefetch(tf.data.AUTOTUNE)

'''


#%%

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)


#%%

# optimizers, loss, metrics

lr = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #Use sparce since these are integer encodings (not one-hot)  #does this need logits=True
metrics = ['accuracy']

# compile the model
net.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

epochs = 10

#%%

net.summary()

#%%
'''
# Training Loop

def train_loop(net_fn, arch, train_iter, test_iter, num_epochs, lr):
    # What is the GPU
    #device_name = tf.device(device="/gpu:0")
    # Use the GPU
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    
    with strategy.scope():
        
        lr = 1e-3
        optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=lr)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy() #Use sparce since these are integer encodings (not one-hot)  #does this need logits=True
        metrics = [tf.keras.metrics.Accuracy(),
                   tf.keras.metrics.AUC()]
        
        # What type of NN are you using
        net = net_fn(arch)
        
        net.compile(optimizer=optimizer, loss=loss_fn, 
                    metrics = metrics)
    
    #callback()  #I don't think I need this to fit
    
    net.fit(train_iter, epochs=num_epochs)
    
    return(net)
    
'''

#%%
'''
# Defined in file: ./chapter_convolutional-neural-networks/lenet.md
class TrainCallback(tf.keras.callbacks.Callback):
    """A callback to visiualize the training progress."""
    def __init__(self, net, train_iter, test_iter, num_epochs, device_name):
        self.timer = dl.Timer()
        self.animator = dl.Animator(
            xlabel='epoch', xlim=[1, num_epochs],
            legend=['train loss', 'train acc', 'test acc'])
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.device_name = device_name

    def on_epoch_begin(self, epoch, logs=None):
        self.timer.start()

    def on_epoch_end(self, epoch, logs):
        self.timer.stop()
        test_acc = self.net.evaluate(self.test_iter, verbose=0,
                                     return_dict=True)['accuracy']
        metrics = (logs['loss'], logs['accuracy'], test_acc)
        self.animator.add(epoch + 1, metrics)
        if epoch == self.num_epochs - 1:
            batch_size = next(iter(self.train_iter))[0].shape[0]
            num_examples = batch_size * tf.data.experimental.cardinality(
                self.train_iter).numpy()
            print(f'loss {metrics[0]:.3f}, train acc {metrics[1]:.3f}, '
                  f'test acc {metrics[2]:.3f}')
            print(f'{num_examples / self.timer.avg():.1f} examples/sec on '
                  f'{str(self.device_name)}')
            
'''           
#%%
'''
# Defined in file: ./chapter_preliminaries/calculus.md
def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')


# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: self.set_axes(self.axes[
            0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
'''

#%%

# Train the model

train_loop(vgg, conv_arch, train_data, val_data, epochs, lr)
    



#%%
#net.build(train_data)
net.summary()

#%%

#train the model  (DOES NOT WORK RIGHT NOW!  SIZE MISSMATCH)

# THIS NEEDS AN ARGMAX LAYER TO GIVE ME THE INDEX OF THE MAX PROB
# FROM THE SOFTMAX FUNCTION.  HOW DO I ADD THAT?



history = net.fit(train_data, 
                  validation_data=val_data, 
                  epochs=epochs)





#%%

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
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

























    