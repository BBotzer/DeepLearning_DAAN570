# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 19:02:29 2023

@author: btb5103 - Brandon Botzer
"""
#%%
import os

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_files
import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras import layers

#%%

dataset, info = tfds.load('imdb_reviews', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

train_dataset.element_spec

#%%
# Set up AUTOTUNING for the data

BUFFER_SIZE = 10000
BATCH_SIZE = 32  # I don't trust my GPU to keep up with 64


# This allows preprocessing to happen on the CPU asynchronously

# QUESTION:  CAN I PUT THIS ANYWHERE ONCE THE DATASET IS CREATED?.......
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


#%%

for example, label in train_dataset.take(1):
  print('texts: ', example.numpy()[:3])
  print()
  print('labels: ', label.numpy()[:3])



#%%

# Text Encoder layer

vocab_size = 2000
max_length = 30

# The TextVectorization layer takes care of setting the vocabulary size
# via max_tokens.  The output_sequence either pads or truncates to the 
# max length prescribed.  The Standardize sets all words to lowercase and 
# removes punctuation.

text_encoder = layers.TextVectorization(max_tokens=vocab_size,
                                        output_sequence_length=max_length,
                                        standardize='lower_and_strip_punctuation'                                        
                                        )

# The .adapt function call 
text_encoder.adapt(train_dataset.map(lambda text, label: text))

# ...... OR DOES THE AUTOTUNING NEED TO GO HERE? AS IN:
    # https://keras.io/guides/preprocessing_layers/

#%%

# This is the vocabulary

vocab = np.array(text_encoder.get_vocabulary())
# Show the first 20
vocab[:20]




#%%

# Proof of functionallity of the text encoder moving text to the indicies

encoded_example = text_encoder(example)[:3].numpy()

encoded_example




#%%

# Show the train and test datasets are balanced in size


print("Length of Training and Test Datasets")
print(len(train_dataset), len(test_dataset), len(label))
print("By 32 batches gives 25,024.  The last batch is not 782 but 758.")


#%%

# Building the RNN with three layers
'''
The SimpleRNN layer with 5 neurons and initialize its kernel with stddev=0.001

The Embedding layer and initialize it by setting the word embedding 
dimension to 50. This means that this layer takes each integer in the 
sequence and embeds it in a 50-dimensional vector.

The output layer has the sigmoid activation function.

'''

# In general the order is:
    # Encoder
    # 




model_RNN = tf.keras.Sequential([
    text_encoder,
    
    
    ])

















