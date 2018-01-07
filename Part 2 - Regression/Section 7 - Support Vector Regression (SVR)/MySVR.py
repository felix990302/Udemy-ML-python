#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 18:55:48 2018

@author: cfzhou
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("Position_Salaries.csv").iloc[:,1:].values



# spliting into training, CV, test sets

X = dataset[:,:-1]
y = dataset[:,-1]

np.random.shuffle(dataset)
X_train = dataset[:, :-1]
y_train = dataset[:, -1]

# feature scaling / mean normalization

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train.reshape(-1,1))

# fit SVR 
from sklearn.svm import SVR
reg = SVR(kernel='rbf', C=10,)
reg.fit(X_train,y_train)

# visualize
X_grid = np.arange(min(X_train),max(X_train),0.001)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_grid,reg.predict(X_grid))
plt.show()

# predict new result with Support Vector regression
print(sc_y.inverse_transform(reg.predict(sc_X.transform(np.array([[6.5]])))))















