# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:04:36 2023

@author: btb51

Questions
Part 1: Data Exploration and Preprocessing

    Read and load data into Python
    Explore and pre-process the dataset. For examples;
        Handle Missing values
        Check Duplicate values
        Outliers detection
        Check correlation
        Check imbalanced data
        Scale or Normalize data
        Plots: Histograms, Boxplots, pairplot, etc.
"""
#PART 1:

#from dl import tensorflow as dl

import pandas as pd
import numpy as np
import requests
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


#I don't think the Pima Indians Diabetes dataset is hosted at the 
#UCI ML Repository anymore.
#I was able to pull it from Kaggle
url = 'https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/download?datasetVersionNumber=1'

file = 'C:\\Users\\btb51\\Documents\\GitHub\\DeepLearning_DAAN570\\DAAN570_Instructor_Sample_Codes\\Lesson_06_Code\\archive\\diabetes.csv'

data = pd.read_csv(file)

#%%

#Turn missing values to NANs with the exception of pregnacies
data["BloodPressure"].replace(to_replace=0, value=np.NAN, inplace=True)
data["SkinThickness"].replace(to_replace=0, value=np.NAN, inplace=True)
data["Insulin"].replace(to_replace=0, value=np.NAN, inplace=True)

#It may be beneficial to_replace with the average of the column if the zeros
#push values

#drop the duplicates keeping the first instance of any dups
data = data.drop_duplicates(keep='first')

#%%
#Check for outliers (keep anything where all data cols are within 3 std dev)
data = data[(np.abs(stats.zscore(data, nan_policy='omit')) < 3).all(axis=1)]


#%%
#Make some quick plots to see if there are any possible imbalances

#It looks like zeros dominate here
fig1, ax1 = plt.subplots()
ax1.hist(data["Pregnancies"])
ax1.set_title("Pregnancies")

#It looks like the age of 20-30 dominates here
fig2, ax2 = plt.subplots()
ax2.hist(data['Age'], bins = 10)
ax2.set_title("Age")

#The outcome looks fair enough, I don't think I need to work on imbalances
fig3, ax3 = plt.subplots()
ax3.hist(data['Outcome'])
ax3.set_title("Outcome")

fig4, ax4 = plt.subplots()
ax4.hist(data['Glucose'])
ax4.set_title('Glucose')



#%%
#Deal with the class imbalance
from imblearn.over_sampling import SMOTE


y = data.iloc[:, 8]
x = data.iloc[:,:8]

oversample = SMOTE()

x, y = oversample.fit_resample(x,y)


print("Length of x: " + str(len(x)))
print("Length of y: " + str(len(y)))


#%%
#Make some quick plots to see if there are any possible imbalances

#It looks like zeros dominate here
fig1, ax1 = plt.subplots()
ax1.hist([x["Pregnancies"]])
ax1.set_title("Pregnancies_balanced")

#It looks like the age of 20-30 dominates here
fig2, ax2 = plt.subplots()
ax2.hist(x['Age'], bins = 10)
ax2.set_title("Age_balanced")

#The outcome looks fair enough, I don't think I need to work on imbalances
fig3, ax3 = plt.subplots()
ax3.hist(y)
ax3.set_title("Outcome_balanced")

fig4, ax4 = plt.subplots()
ax4.hist(x['Glucose'])
ax4.set_title('Glucose_balanced')



#%%
#Check correlation
cor = data.corr()

# plotting
plt.matshow(cor)
plt.colorbar()
plt.xticks(ticks=range(9),labels="")
plt.yticks(ticks=range(9),labels = data.columns)

#Pairplot
sns.pairplot(data)



#%%
#Scale/Normalize the data

#use minmaxscaler
scaler = MinMaxScaler()


data_x = scaler.fit_transform(x)



#%%

#PART 2:
"""Part 2: Build a Baseline Model

Use the Sequential model to quickly build a baseline neural network with one single hidden layer with 12 nodes.

    Split the data to training and testing dataset (75%, 25%)
    Build the baseline model and find how many parameters does your model have?
    Train you model with 20 epochs with RMSProp at a learning rate of .001 and a batch size of 128
    Graph the trajectory of the loss functions, accuracy on both train and test set.
    Evaluate and interpret the accuracy and loss performance during training, and testing.
"""
    
#%%

#Model imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import plot_model
from keras.layers import concatenate
from keras.metrics import AUC

from sklearn.model_selection import train_test_split


#%%

#train test split
X_train, X_test, y_train, y_test = train_test_split(data_x, y, test_size=0.25,
                                                    random_state=570)

#%%
# Model Generation

def build_model():
    
    model_input = Input(shape=(8,), name='data_in')
    hidden_layer_1 = Dense(units=12, activation='relu', name='HL_1')(model_input)
    model_out = Dense(1, activation='sigmoid', name='data_out')(hidden_layer_1)
    
    model = Model(inputs=model_input, outputs=model_out, name='Diabetes')
    
    return model



#%%

##Compile the model
model = build_model()

#USING RMSProp
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

bi_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

metric = [tf.keras.metrics.BinaryAccuracy(),
          tf.keras.metrics.FalsePositives(),
          tf.keras.metrics.AUC(curve='ROC')]

model.compile(optimizer=optimizer, loss=bi_loss, metrics=metric)


#%%

# Train the model

values = model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1)


#Evaluate the model
loss, accuracy, false_pos, auc = model.evaluate(X_test, y_test)


#%%
# Model Summary statistics and figure
print(model.summary())

plot_model(model, to_file='Diabetes_Model_V1.png')

#%%

def quick_plot(values, keys,title = "You Need A Title"):
    
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
quick_plot(values, list(values.history.keys()),"RMSProp, Batch=128, lr=0.001, epochs=20, activ=relu/sigmoid")






#%%
#PART 3:
"""Part 3: Find the Best Model

Now try four different models and see if you can improve the accuracy by 
focusing on different network structures 
(i.e, activation functions, optimization algorithms, batch sizes, 
 number of epochs, ...), affecting the performance, 
training time, and level of overfitting (or underfitting).

    For all your models, plot the ROC curve for the predictions.
    Which model has best performance, why?
    Save your best model weights into a binary file.

Submit two files: the Jupyter notebook with your code and answers and its print out PDF.
"""

#%%

#Model V2  (New Structure in General)
# Using Adam as Optimizer; lr is 0.01; batchsize = 41; epochs = 100


def build_model_v2():
    
    model_input = Input(shape=(8,), name='data_in')
    hidden_layer_1 = Dense(units=12, activation='sigmoid', name='HL_1')(model_input)
    hidden_layer_2 = Dense(units=12, activation='sigmoid', name='HL_2')(hidden_layer_1)
    hidden_layer_3 = Dense(units=12, activation='sigmoid', name='HL_3')(hidden_layer_2)
    model_out = Dense(1, activation='sigmoid', name='data_out')(hidden_layer_3)
    
    model = Model(inputs=model_input, outputs=model_out, name='Diabetes')
    
    return model



#%%

##Compile the model
model_v2 = build_model_v2()

#Using ADAM
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

bi_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

metric = [tf.keras.metrics.BinaryAccuracy(),
          tf.keras.metrics.FalsePositives(),
          tf.keras.metrics.AUC(curve='ROC')]

model_v2.compile(optimizer=optimizer, loss=bi_loss, metrics=metric)


#%%

# Train the model

values_v2 = model_v2.fit(X_train, y_train, batch_size=41, epochs=100, verbose=1)


#Evaluate the model
loss, accuracy, false_pos, auc = model_v2.evaluate(X_test, y_test)

#%%

# Model Summary statistics and figure
print(model_v2.summary())

plot_model(model_v2, to_file='Diabetes_Model_v2.png')

quick_plot(values_v2, list(values_v2.history.keys()),"Adam; batch 41, lr= 0.01; epochs = 100, sigmoids")



##############################################################################
#%%

# Model V3

# Origional Structure; More Epochs as only change

#%%
# Model Generation

def build_model_v3():
    
    model_input = Input(shape=(8,), name='data_in')
    hidden_layer_1 = Dense(units=12, activation='relu', name='HL_1')(model_input)
    model_out = Dense(1, activation='sigmoid', name='data_out')(hidden_layer_1)
    
    model = Model(inputs=model_input, outputs=model_out, name='Diabetes')
    
    return model



#%%

##Compile the model
model_v3 = build_model()

#USING RMSProp
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

bi_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

metric = [tf.keras.metrics.BinaryAccuracy(),
          tf.keras.metrics.FalsePositives(),
          tf.keras.metrics.AUC(curve='ROC')]

model_v3.compile(optimizer=optimizer, loss=bi_loss, metrics=metric)


#%%

# Train the model

values_v3 = model_v3.fit(X_train, y_train, batch_size=128, epochs=500, verbose=1)


#Evaluate the model
loss, accuracy, false_pos, auc = model_v3.evaluate(X_test, y_test)


#%%
# Model Summary statistics and figure
print(model_v3.summary())

plot_model(model_v3, to_file='Diabetes_Model_V3.png')

quick_plot(values_v3, list(values_v3.history.keys()),"RMSProp, Batch=128, lr=0.001, epochs=500, activ=relu/sigmoid")



#%%##########################################################################
# Model V4

# Stochastic Gradient Decent; Stochastic (sample by sample)


#%%
# Model Generation

def build_model_v4():
    
    model_input = Input(shape=(8,), name='data_in')
    hidden_layer_1 = Dense(units=12, activation='relu', name='HL_1')(model_input)
    model_out = Dense(1, activation='sigmoid', name='data_out')(hidden_layer_1)
    
    model = Model(inputs=model_input, outputs=model_out, name='Diabetes')
    
    return model



#%%

##Compile the model
model_v4 = build_model()

#USING RMSProp
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

bi_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

metric = [tf.keras.metrics.BinaryAccuracy(),
          tf.keras.metrics.FalsePositives(),
          tf.keras.metrics.AUC(curve='ROC')]

model_v4.compile(optimizer=optimizer, loss=bi_loss, metrics=metric)


#%%

# Train the model

values_v4 = model_v4.fit(X_train, y_train, batch_size=1, epochs=506, verbose=1)


#Evaluate the model
loss, accuracy, false_pos, auc = model_v4.evaluate(X_test, y_test)


#%%
# Model Summary statistics and figure
print(model_v4.summary())

plot_model(model_v4, to_file='Diabetes_Model_V4.png')

quick_plot(values_v4, list(values_v4.history.keys()),"SGD, Batch=1, lr=0.001, epochs=506, activ=relu/sigmoid")


#Model 4 runs SLOWLY!  ~1 to 2 seconds per epoch




























































