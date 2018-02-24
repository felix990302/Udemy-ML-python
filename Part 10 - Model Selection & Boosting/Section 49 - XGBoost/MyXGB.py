#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 23:52:20 2018

@author: cfzhou
"""


# importing libraries
import numpy as np
import matplotlib.pyplot as plt
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


# Fitting XGBoost
from xgboost import XGBClassifier
clas = XGBClassifier()
clas.fit(X_train, y_train)


# predict test set
y_pred = clas.predict(X_test)

# scoring

y_pred[y_pred<0.5]=0
y_pred[y_pred>=0.5]=1

from sklearn import metrics
final_score = metrics.f1_score(y_test, y_pred, average='micro')








