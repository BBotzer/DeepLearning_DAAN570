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

# Optuna imports
import optuna
from keras.backend import clear_session

#%%

dataset, info = tfds.load('imdb_reviews', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

train_dataset.element_spec

#%%
# Set up AUTOTUNING for the data

BUFFER_SIZE = 4000
BATCH_SIZE =  30 #(text, label) pairs


# This allows preprocessing to happen on the CPU asynchronously

# QUESTION:  CAN I PUT THIS ANYWHERE ONCE THE DATASET IS CREATED?.......
train_dataset_30 = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset_30 = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


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
    # Embedding
    # SimpleRNN
    # Dense Output layer
    
# If you put the SimpleRNN before the Embedding you'll get a Input error
# as the shape of the tensor is wrong.


# I think I need to put the text_encoder at the begining of this 
# so that the text is in numeric values

# HOW DO I SET-UP THE KERNEL INITIALIZATION CORRECTLY?



def model_RNN(encoder, seq_length:int):
    
    model = tf.keras.Sequential([
        encoder,
        layers.Embedding(input_dim=vocab_size, output_dim=seq_length),
        layers.SimpleRNN(5, kernel_initializer='glorot_uniform'), # setting return_sequences=True will kick back the batch size as well
        layers.Dense(1, activation='sigmoid')
        ])
    
    return model

#%%
# Create the network
rnn_net = model_RNN(encoder=text_encoder, seq_length=30)

rnn_net.summary()

print("By the summary, there are 100,000 parameters in the embedding layer.")

#%%

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

mets = ['accuracy']

rnn_net.compile(optimizer, loss_fn, mets)


#%%
history_rnn = rnn_net.fit(train_dataset_30, 
                          epochs=3,  # Change to 10 later
                          validation_data=test_dataset_30,
                          validation_steps=30)


#%%

test_loss, test_acc = rnn_net.evaluate(test_dataset)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)




#%%

# Plotting Function
def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

#%%
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history_rnn, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history_rnn, 'loss')
plt.ylim(0, None)

#%%
















#%%

# Tunning
'''

11- Prepare the data to use sequences of length 80 rather than length 30 and 
retrain your model. Did it improve the performance?

12- Try different values of the maximum length of a sequence ("max_features").
 Can you improve the performance?

13- Try smaller and larger sizes of the RNN hidden dimension. 
How does it affect the model performance? How does it affect the run time?
'''

# Set up AUTOTUNING for the data

BUFFER_SIZE = 4000
BATCH_SIZE =  80 #(text, label) pairs


# This allows preprocessing to happen on the CPU asynchronously

# QUESTION:  CAN I PUT THIS ANYWHERE ONCE THE DATASET IS CREATED?.......
train_dataset_80 = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset_80 = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# The .adapt function call 
# This is the encoder with max output sequence of 30... Should that have changed?
text_encoder.adapt(train_dataset_80.map(lambda text, label: text))


#%%
for example, label in train_dataset_80.take(1):
  print('texts: ', example.numpy()[:3])
  print()
  print('labels: ', label.numpy()[:3])
  
encoded_example = text_encoder(example)[:3].numpy()

encoded_example
  

#%%
rnn_net = model_RNN(encoder=text_encoder, seq_length=30)


rnn_net.compile(optimizer, loss_fn, mets)


history_rnn_2 = rnn_net.fit(train_dataset_80, 
                          epochs=3,  # Change to 10 later
                          validation_data=test_dataset_80,
                          validation_steps=30)

#%%

#Pulling an error right now...
'''
Node: 'sequential_6/text_vectorization/StringSplit/StringSplitV2'
2 root error(s) found.
  (0) INVALID_ARGUMENT:  input must be a vector, got shape: []
	 [[{{node sequential_6/text_vectorization/StringSplit/StringSplitV2}}]]
	 [[Func/binary_crossentropy/cond/then/_43/binary_crossentropy/cond/cond/then/_128/binary_crossentropy/cond/cond/remove_squeezable_dimensions/cond_1/else/_245/input/_281/_100]]
  (1) INVALID_ARGUMENT:  input must be a vector, got shape: []
	 [[{{node sequential_6/text_vectorization/StringSplit/StringSplitV2}}]]
0 successful operations.
0 derived errors ignored. [Op:__inference_test_function_51883]


2023-07-12 18:32:48.890862: W tensorflow/core/framework/op_kernel.cc:1780] OP_REQUIRES failed at strided_slice_op.cc:111 : INVALID_ARGUMENT: slice index 0 of dimension 0 out of bounds.
'''

test_loss_2, test_acc_2 = rnn_net.evaluate(test_dataset)

print('Test Loss:', test_loss_2)
print('Test Accuracy:', test_acc_2)

#%%
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history_rnn_2, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history_rnn_2, 'loss')
plt.ylim(0, None)




#%%
# Text Encoder layer is now changed to go to 80 words

BUFFER_SIZE = 4000
BATCH_SIZE =  80 #(text, label) pairs

vocab_size = 2000
max_length = 80  # Changed sequence length

# The TextVectorization layer takes care of setting the vocabulary size
# via max_tokens.  The output_sequence either pads or truncates to the 
# max length prescribed.  The Standardize sets all words to lowercase and 
# removes punctuation.


text_encoder2 = layers.TextVectorization(max_tokens=vocab_size,
                                        output_sequence_length=max_length,
                                        standardize='lower_and_strip_punctuation'                                        
                                        )

train_dataset_80_v2 = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset_80_v2 = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# The .adapt function call 
# This is the encoder with max output sequence of 30... Should that have changed?
text_encoder2.adapt(train_dataset_80_v2.map(lambda text, label: text))



#%%
for example, label in train_dataset_80_v2.take(1):
  print('texts: ', example.numpy()[:3])
  print()
  print('labels: ', label.numpy()[:3])
  
encoded_example = text_encoder(example)[:3].numpy()

encoded_example


#%%

# Create the network
rnn_net80 = model_RNN(encoder=text_encoder2, seq_length=80)

rnn_net80.summary()

print("By the summary, there are 160,436 parameters in the embedding layer.")




optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

mets = ['accuracy']

rnn_net80.compile(optimizer, loss_fn, mets)


#%%
history_rnn80 = rnn_net80.fit(train_dataset, 
                          epochs=3,  # Change to 10 later
                          validation_data=test_dataset,
                          validation_steps=30)


#%%

test_loss80, test_acc80 = rnn_net80.evaluate(test_dataset)

print('Test Loss:', test_loss80)
print('Test Accuracy:', test_acc80)

#%%
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history_rnn80, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history_rnn80, 'loss')
plt.ylim(0, None)
#%%

# NEW RNN with modified hidden dimension

def objective_hiddenlayer_rnn(trial):
    
    # Free up memory and session hisotry    
    clear_session()
    
    # Load Data
    dataset, info = tfds.load('imdb_reviews', with_info=True,
                              as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    train_dataset.element_spec

    vocab_size = 2000
    seq_length = 80
    
    # This allows preprocessing to happen on the CPU asynchronously
    BUFFER_SIZE = 4000
    BATCH_SIZE =  80 #(text, label) pairs
    # QUESTION:  CAN I PUT THIS ANYWHERE ONCE THE DATASET IS CREATED?.......
    # Can I add the num_parallel_calls here?
    # Should I be calling catch() here?
    train_dataset_obj = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset_obj = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


    # Encode the text to numbers
    text_encoder_obj = layers.TextVectorization(max_tokens=vocab_size,
                                            output_sequence_length=max_length,
                                            standardize='lower_and_strip_punctuation'                                        
                                            )

    # The .adapt function call 
    text_encoder_obj.adapt(train_dataset_obj.map(lambda text, label: text))
    


    # Create the trial parameter for the number of hidden layers
    n_hidden = trial.suggest_categorical("n_hidden", [1,2,3,4,5,6,7,8,9,10])
    
    # Create the trial parameter for the output_dimension
    #n_output_dim = trial.suggest_categorical('n_out_dim', [30,50,80])
    
    # Cre
    
    
    # Begin model creation #########################

    
    model = tf.keras.Sequential([
        text_encoder,
        layers.Embedding(input_dim=vocab_size, output_dim=seq_length),
        layers.SimpleRNN(n_hidden, kernel_initializer='glorot_uniform'), # setting return_sequences=True will kick back the batch size as well
        layers.Dense(1, activation='sigmoid')
        ])
    
    # End model Creation #########################
    
    # Begin training
    
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    mets = ['accuracy']

    model.compile(optimizer, loss_fn, mets)
    
    history = model.fit(train_dataset_obj, 
                        epochs = 20, # Change later
                        validation_data=test_dataset_obj,                        
                        validation_steps=30
                        )
    
    eval_score = model.evaluate(test_dataset_obj, verbose=0)
    
    return eval_score[1]

#%%

# Create Study

study = optuna.create_study(direction='maximize')

study.optimize(objective_hiddenlayer_rnn, n_trials = 25, timeout = 16000, 
               gc_after_trial=True)


#%%
# Print outputs of study

print("You completed {} trials.".format(len(study.trials)))

print("Best run:")
best = study.best_trial
print("    Accuracy Value: {}".format(best.value))

# Now for the parameters
print("    Parameters: ")
for key, value in best.params.items():    # go through the dictionary
    print("        {}: {}".format(key,value))

'''

You completed 25 trials.
Best run:
    Accuracy Value: 0.7139999866485596
    Parameters: 
        n_hidden: 3

'''    


#%%
'''

Chaning the hidden dimension from 1 to 10 did not change my epoch runtime
by any considerable margin.  Each epoch was around 24 -26 seconds.

However, I am uncertain if I am being bottlenecked by my CPU as my
GPU is only working at 6%... I would have expected my GPU to be at least
at 40-50% when training and I am afraid I may be catching the data 
incorrectly.

'''
        
 
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


# Assign the GPU to a variable
# GPU is also known as "/device:GPU:0"
dev = try_gpu()

# COuld I use this in the Optuna section to ensure GPU usage??
# https://www.tensorflow.org/api_docs/python/tf/distribute/OneDeviceStrategy
# https://www.tensorflow.org/tutorials/distribute/save_and_load
# strategy = tf.distribute.OneDeviceStrategy(device=dev)
# with strategy.scope():
    #DO THE MODEL CREATION
    #COMPILE THE MODEL
    #Do THE MODEL FIT




#%%

'''
IF I HAVE TIME, I'D LIKE TO GO BACK AND LOOK AT THE PERPLEXITY METRIC
https://keras.io/api/keras_nlp/metrics/perplexity/
'''



#%%

# GRU Model Build
'''

gru_cell = tf.keras.layers.GRUCell(num_hiddens,
                                   kernel_initializer='glorot_uniform')
gru_layer = tf.keras.layers.RNN(gru_cell, time_major=True,
                                return_sequences=True, return_state=True)

device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    model = d2l.RNNModel(gru_layer, vocab_size=len(vocab))

train(model, train_iter, vocab, lr, num_epochs, strategy)

'''



def model_GRU():
    
    gru_cell = layers.GRUCell()
    
    gru_layer = layers.RNN(gru_cell, )












#%%

# LSTM Model Build

'''

lstm_cell = tf.keras.layers.LSTMCell(num_hiddens,
                                     kernel_initializer='glorot_uniform')
lstm_layer = tf.keras.layers.RNN(lstm_cell, time_major=True,
                                 return_sequences=True, return_state=True)
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    model = RNNModel(lstm_layer, vocab_size=len(vocab))
train(model, train_iter, vocab, lr, num_epochs, strategy)

'''


def model_LSTM():
    
        lstm_cell = layers.LSTMCell()
        
        lstm_layer = layers.RNN(lstm_cell)























































