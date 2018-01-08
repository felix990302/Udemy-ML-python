#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 19:57:17 2018

@author: cfzhou
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("Position_Salaries.csv").iloc[:, 1:].values

# spliting into training, CV, test sets
train = dataset
np.random.shuffle(train)

X_train = train[:, :-1]
y_train = train[:, -1]

# fit decision tree regressor
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(criterion='friedman_mse', presort=True)
reg.fit(X_train, y_train)

# visualize
X_grid = np.arange(min(X_train),max(X_train),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_grid,reg.predict(X_grid))
plt.show()

# predict new result with Support Vector regression
print(reg.predict(np.array([[6.5]])))








