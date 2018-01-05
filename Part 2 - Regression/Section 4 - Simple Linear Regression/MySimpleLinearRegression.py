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
dataset = pd.read_csv("Salary_Data.csv").iloc[:, :].values

# visualize
# dataset.plot(kind='scatter', x=0, y=1, figsize=(12,8))

# spliting into training, CV, test sets
dataset = pd.DataFrame(dataset)
train, test = np.split(dataset.sample(frac=1), [int(.8 * len(dataset))])
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values
# X_cv = cv.iloc[:, :-1].values
# y_cv = cv.iloc[:, -1].values
X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

# fitting data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# test set accuracy
y_pred = regressor.predict(X_test)

# Visualizing
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Years of Experience (training set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')

