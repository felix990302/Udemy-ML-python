#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 19:23:23 2018

@author: cfzhou
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
orig = pd.read_csv("Social_Network_Ads.csv").iloc[:, :].values
dataset = orig[:,1:]

# encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
dataset[:, 0] = labelencoder.fit_transform(dataset[:, 0])

onehotencoder = OneHotEncoder(categorical_features=[0])

dataset = onehotencoder.fit_transform(dataset).toarray()[:,1:]

# spliting into training, CV, test sets
np.random.shuffle(dataset)
y_train, y_cv, y_test = np.split(dataset, [int(.6 * len(dataset)), int(.8 * len(dataset))])
X_train = y_train[:, :-1]
y_train = y_train[:, -1]
X_cv = y_cv[:, :-1]
y_cv = y_cv[:, -1]
X_test = y_test[:, :-1]
y_test = y_test[:, -1]

# feature scaling / mean normalization

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_cv = sc_X.transform(X_cv)
X_test = sc_X.transform(X_test)

