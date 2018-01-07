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
dataset = pd.read_csv("Position_Salaries.csv").iloc[:,1:].values

# spliting into training, CV, test sets
dataset = pd.DataFrame(dataset)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

train = dataset.sample(frac=1)
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values

# fit linear model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

# fit polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly_feat = PolynomialFeatures(degree = 4)
X_poly = poly_feat.fit_transform(X_train)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y_train)

# visualize/compare both
X_grid = np.arange(min(X),max(X),0.001)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_grid,lin_reg.predict(X_grid))
plt.plot(X_grid ,poly_reg.predict(poly_feat.fit_transform(X_grid)), color='green')
plt.show()

# Predict new result with lin / poly regression
lin_reg.predict(6.5)

poly_reg.predict(poly_feat.fit_transform(6.5))

