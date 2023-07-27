# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 20:30:33 2023

@author: btb5103 - Brandon Botzer

"""

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# IMPORT AND LOAD MODULES

import os, shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import keras.utils as image

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# CHECK FOR GPU

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

device = try_gpu()  # will return an _EagerDeviceContext

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# UTILTIY FUNCTIONS

# Load Images and Visualize
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


# Image augumentation - it takes image augumentation operation as an argument and applies to image
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)


#%%


# A Generic plotter provided by the class

def plot_acc_metric(history, epochs, title='You need a title...'):
    '''Basic plot for Accuracy and Training Loss

    Args:
        history: the history from model.fit [history = model.fit(....)]
        epochs: the number of epochs the fit was run for
        title: Don't forget your title
    '''

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

# A generic 4 plot plotter
# Usefull for plotting [Accuracy, F1, Loss, ROC graph

def quick_plot(history, keys, title = "You Need A Title"):
    '''Plot for 2x2 plot of items based on the model.fit history

    Args:
        history: the history from model.fit [history = model.fit(....)]
        keys: Provided typically from the history as --> list(values.history.keys())
        title: Don't forget your title
    '''

    # Plot loss function of the training
    fig, axs = plt.subplots(2,2)
    fig.suptitle(title)
    fig.tight_layout()
    
    axs[0,0].plot(values.history[keys[0]])
    axs[0,0].set_ylabel('loss')
    axs[0,0].set_xlabel('epoch')
    
    axs[0,1].plot(values.history[keys[1]])
    axs[0,1].set_ylabel(keys[1])
    axs[0,1].set_xlabel('epoch')
    
    axs[1,0].plot(values.history[keys[2]])
    axs[1,0].set_ylabel(keys[2])
    axs[1,0].set_xlabel('epoch')
    
    axs[1,1].plot(values.history[keys[3]])
    axs[1,1].set_ylabel(keys[3])
    axs[1,1].set_xlabel('epoch')

#%%

# Explore Dimensions of our data
# May not be needed given our data should be 50x50 3 channel images
def explore_dimensions(path, labels):
    # credit to Suradech Kongkiatpaiboon
    import seaborn as sns
    #Explore the avearage dimension of the images
    dim1 = []
    dim2 = []
    for image_filename in os.listdir(path+'//'+labels):    
      
        img = imread(path+'//'+labels+'//'+image_filename)
        if len(img.shape)==2: #Reshape some images with single color channel
            img = img.reshape(img.shape[0],img.shape[1],1)
        d1,d2,colors = img.shape
        dim1.append(d1)
        dim2.append(d2)
    p = sns.jointplot(dim1,dim2)
    p.fig.suptitle("Dimensions of "+labels+ " images")
    
    
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# DATA LOAD

# Use ImageDataGenerator for preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Enter Data Path based on your machine and data location
data_path = 'C:/Users/btb51/Documents/GitHub/BreastCancerImages/breast-histopathology-images/data' 
#data_path = './data/' # Used for DeepNote

#%%
# Unaugmented Images

# This is a # use keras.util.image_dataset_from_directory to import all the image files
def data_build(batchsize=32, i_h=50, i_w=50, filepath=data_path):

    '''Load in data from a directory with set categories to be 
        interpreted by system.
    Args:
        batchsize: size of each batch
        i_h: Image height to resize to [50 for the B.C. database]
        i_w: Image width to resize to [50 for the B.C. database]
        filepath: Top layer directory path at which the categorical data resides. 
            ie. data:
                    Benign
                    Malignant  
    '''
    #datasets will be a tuple of the train and validation data
    train_ds = image_dataset_from_directory(filepath,
                                batch_size=batchsize,
                                image_size = (i_h,i_w),  
                                shuffle = True,
                                seed = 570,
                                validation_split = 0.2,
                                subset = 'training')


    val_ds = image_dataset_from_directory(filepath,
                                batch_size=batchsize,
                                image_size = (i_h,i_w),  
                                shuffle = True,
                                seed = 570,
                                validation_split = 0.2,
                                subset = 'validation')

    return train_ds, val_ds
#%%
# Collect the data
train_ds, val_ds = data_build(32,50,50,data_path)


#%%
# Augmented Images Setup

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rescale=1./255, 
    validation_split=0.2, # keep out 0.20 for testing
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(
    rescale=1 / 255.0,
    validation_split=0.2)


#%% Load Data
# Adjust based on GPU specs
batchsize = 32 

trainGen = trainAug.flow_from_directory(
	data_path,
	class_mode="categorical",
	target_size=(50, 50),
	color_mode="rgb",
	shuffle=True,
	batch_size=batchsize,
    subset='training')

# initialize the validation generator
valGen = valAug.flow_from_directory(
	data_path,
	class_mode="categorical",
	target_size=(50, 50),
	color_mode="rgb",
	shuffle=True,
	batch_size=batchsize,
    subset='validation')

    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
# SOME INITIAL IMAGES

# change directory

os.chdir('C:/Users/btb51/Documents/GitHub/BreastCancerImages/breast-histopathology-images/')

histpatho_img = glob.glob('./data/*/*.png', recursive = True)

for imgname in histpatho_img[:6]:
    print(imgname)

#%%

# Plot initial set of benign and malignant images

N_IDC = []
P_IDC = []

for img in histpatho_img:
    if img[-5] == '0' :
        N_IDC.append(img)
    
    elif img[-5] == '1' :
        P_IDC.append(img)
plt.figure(figsize = (15, 15))

some_non = np.random.randint(0, len(N_IDC), 18)
some_can = np.random.randint(0, len(P_IDC), 18)

s = 0
for num in some_non:
    
        img = image.load_img((N_IDC[num]), target_size=(100, 100))
        img = image.img_to_array(img)
        
        plt.subplot(6, 6, 2*s+1)
        plt.axis('off')
        plt.title('Benign')
        plt.imshow(img.astype('uint8'))
        s += 1
s = 1
for num in some_can:
    
        img = image.load_img((P_IDC[num]), target_size=(100, 100))
        img = image.img_to_array(img)
        
        plt.subplot(6, 6, 2*s)
        plt.axis('off')        
        plt.title('Malignant')
        plt.imshow(img.astype('uint8'))
        s += 1


#%%

# Flipping the image left and right 
img = plt.imread('./data/benign/10253_idx5_x1401_y1001_class0.png')
apply(img, tf.image.random_flip_left_right)


#%%

# Flip image up and down 
apply(img, tf.image.random_flip_up_down)

#%%

# Randomly changing the hue of the image 
# code from lesson 11 - Image Augmentation
aug=tf.image.random_hue
num_rows=2
num_cols=4
scale=1.5
max_delta=0.5

Y = [aug(img, max_delta) for _ in range(num_rows * num_cols)]
show_images(Y, num_rows, num_cols, scale)
plt.show

#%%

# Randomly changing image brightness 
# code from lesson 11 - Image Augmentation

aug=tf.image.random_brightness
num_rows=2
num_cols=4
scale=1.5
max_delta=0.5

Y = [aug(img, max_delta) for _ in range(num_rows * num_cols)]
show_images(Y, num_rows, num_cols, scale)
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Create DNNs

    # Custom CNN
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization

from keras.layers.convolutional import SeparableConv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense

# Custom CNN Code

def create_custom_CNN(width, height, depth, classes):
    net = Sequential()
    shape =(height, width, depth)

    net.add(SeparableConv2D(32, (3,3), padding='same', input_shape=shape))
    net.add(Activation('relu'))
    net.add(BatchNormalization())
    net.add(MaxPooling2D(pool_size=(2,2)))
    net.add(Dropout(0.23))
        
    net.add(SeparableConv2D(64,(3,3), padding='same'))
    net.add(Activation('relu'))
    net.add(BatchNormalization())
    net.add(SeparableConv2D(64,(3,3), padding='same'))
    net.add(Activation('relu'))
    net.add(BatchNormalization())
    net.add(MaxPooling2D(pool_size=(2,2)))
    net.add(Dropout(0.25))
        
    net.add(SeparableConv2D(128, (3,3), padding='same'))
    net.add(Activation('relu'))
    net.add(BatchNormalization())
    net.add(SeparableConv2D(128, (3,3), padding='same'))
    net.add(Activation('relu'))
    net.add(BatchNormalization())
    net.add(SeparableConv2D (128, (3,3), padding='same'))
    net.add(Activation('relu'))
    net.add(BatchNormalization())
    net.add(MaxPooling2D(pool_size=(2,2)))
    net.add(Dropout(0.25))
        
    net.add(Flatten())
    net.add(Dense(256))
    net.add(Activation('relu'))
    net.add(BatchNormalization())
    net.add(Dropout(0.5))
        
    net.add(Dense(classes))
    net.add(Activation('sigmoid'))
        
    return net

#%%
# Create the custom CNN
cust_cnn = create_custom_CNN(width=50, height=50, depth=3, classes=1)
cust_cnn.summary()   



#%%

    # Pre-trained VGG-19

#from tensorflow.keras.applications.vgg19 import VGG19
# Load VGG19 Pre-trained Model

from keras.applications.vgg19 import VGG19
# load model
vgg = VGG19()
# summarize the model
vgg.summary() 

#%%

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
# I think the images may need to be scaled up to 224x224 since that is what VGG expects
IMAGE_SIZE = [224, 224]
# Load VGG19 network. Create a VGG19 model, and removing the last layer that is classifying 1000 images.  
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False) 

# This sets the base that the layers are not trainable. 
# If we'd want to train the layers with custom data, these two lines can be ommitted. 
for layer in vgg.layers:
    layer.trainable = False

'''
# Create a new top layer to the vgg which will have a similar FC layer to our custom_cnn architecture
model_top = vgg_base.output
model_top = layers.Flatten(name='flatten')(model_top)
model_top = layers.Dense(256, activation='relu')(model_top)
model_top = layers.Dense(128, activation='relu')(model_top)
model_top = BatchNormalization()(model_top)
model_top = Dropout(0.5)(model_top)
output = Dense(1, activation='sigmoid')(model_top) # using a sigmoid with one output for faster updates
'''

x = Flatten()(vgg.output) #Output obtained on vgg19 is now flattened. 
output = Dense(1, activation='sigmoid')(x) # We have 2 classes #probably should just use a sigmoid

# Creating model object 
vgg_plus = tf.keras.Model(inputs=vgg.input, outputs=output) 

vgg_plus.summary()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # COMPILE MODELS

# Setup the model compile items

# Via https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#%%
# Use Adam Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
#Binary Crossentropy for two classes (CHECK from_logits=False)  
# I think it is True from the sigmoid logits
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)  # Check the logits=False ?
# Metrics to compute during training
metrics=[
    tf.keras.metrics.BinaryAccuracy(),  # Accuracy
    tf.keras.metrics.AUC(curve='ROC', from_logits=False),  # Must match loss from_logits
    f1_m,
    tf.keras.metrics.TruePositives()]

#Compile models
cust_cnn.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

vgg.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

vgg_plus.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # CREATE EARLY STOP AND CHECKPOINT CALLBACK
    
# Use early stopping to avoid training while accuracy starts to decrease 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = tf.keras.callbacks.EarlyStopping(monitor='binary_accuracy', patience=5)

# Use a model checkpoint to prevent lossing model due to power issues
custcnn_check_path = 'C:/Users/btb51/Documents/GitHub/DeepLearning_DAAN570/DAAN570_Instructor_Sample_Codes/FinalProject/custcnn_checkpt/models/custcnn_checkpt'
vgg_check_path = 'C:/Users/btb51/Documents/GitHub/DeepLearning_DAAN570/DAAN570_Instructor_Sample_Codes/FinalProject/custcnn_checkpt/models/./vgg_checkpt'
vggplus_check_path = 'C:/Users/btb51/Documents/GitHub/DeepLearning_DAAN570/DAAN570_Instructor_Sample_Codes/FinalProject/custcnn_checkpt/models/./vggplus_checkpt'


    
cnn_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=custcnn_check_path,
    save_weights_only=False,
    save_freq=10,
    # monitor='val_accuracy',
    monitor='binary_accuracy',
    mode='max',
    save_best_only=True)


vgg_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=vgg_check_path,
    save_weights_only=False,
    save_freq=10,
    # monitor='val_accuracy',
    monitor='binary_accuracy',
    mode='max',
    save_best_only=True)
   

vggplus_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=vggplus_check_path,
    save_weights_only=False,
    save_freq=10,
    # monitor='val_accuracy',
    monitor='binary_accuracy',
    mode='max',
    save_best_only=True)


    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    
    # TRAIN VARIOUS MODELS
    
# Train CustCNN model using unaugmented images

# Create, Compile, Fit
cust_cnn_unaug=create_custom_CNN(width=50, height=50, depth=3, classes=1)
cust_cnn_unaug.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

history_custcnn_unaug = cust_cnn_unaug.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs = 20, 
    callbacks=[early_stop, cnn_checkpoint])
    
    
#%%  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    