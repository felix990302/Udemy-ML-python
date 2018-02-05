#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 20:13:15 2018

@author: cfzhou
"""

# importing libraries
import pandas as pd

# import dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values


# encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()[:,1:] # remove linearly dependant entry


# split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# feature scaling / mean normalization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Import Keras
from keras.models import Sequential
from keras.layers import Dense

# Intialize ANN
clas = Sequential()

# Input layer, first hidden layer
clas.add(Dense(activation="relu", 
               input_dim=11, 
               units=6, 
               kernel_initializer="uniform"))

# Add Alot of Layers
layers = 10
for i in range(0, layers):
    clas.add(Dense(activation="relu", 
                   units=6, 
                   kernel_initializer="uniform"))

# Output Layer
clas.add(Dense(activation="sigmoid", 
               units=1, 
               kernel_initializer="uniform"))

# Compile ANN
clas.compile(optimizer = 'adam', 
             loss='binary_crossentropy',
             metrics=['accuracy'])

# Train
clas.fit(X_train, y_train, batch_size=100, epochs=500)


# predict test set
y_pred = clas.predict(X_test)

# scoring

y_pred[y_pred<0.5]=0
y_pred[y_pred>=0.5]=1

from sklearn import metrics
print("\ntest set score:", 
      metrics.f1_score(y_test, y_pred, average='micro'))


# Save model
clas.save('LooANN.h5')










