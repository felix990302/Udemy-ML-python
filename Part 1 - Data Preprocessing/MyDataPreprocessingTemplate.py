#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 20:52:09 2018

@author: cfzhou
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("Data.csv").iloc[:, :].values

# missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
dataset[:, 1:3] = imputer.fit_transform(dataset[:, 1:3])

# encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
dataset[:, 0] = labelencoder.fit_transform(dataset[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
dataset[:, 3] = labelencoder.fit_transform(dataset[:, 3])

dataset = onehotencoder.fit_transform(dataset).toarray()

# spliting into training, CV, test sets
dataset = pd.DataFrame(dataset)
train, cv, test = np.split(dataset.sample(
    frac=1), [int(.6 * len(dataset)), int(.8 * len(dataset))])
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, len(train) - 1].values
X_cv = cv.iloc[:, :-1].values
y_cv = cv.iloc[:, len(cv) - 1].values
X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, len(test) - 1].values

# feature scaling / mean normalization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_cv = sc_X.transform(X_cv)
X_test = sc_X.transform(X_test)
