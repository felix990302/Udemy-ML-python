#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:31:37 2018

@author: cfzhou
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("Position_Salaries.csv").iloc[:, :].values

# missing data
'''
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
dataset[:, 1:3] = imputer.fit_transform(dataset[:, 1:3])
'''

# encode categorical data
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
dataset[:, 0] = labelencoder.fit_transform(dataset[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
dataset[:, 3] = labelencoder.fit_transform(dataset[:, 3])

dataset = onehotencoder.fit_transform(dataset).toarray()
'''

# spliting into training, CV, test sets
np.random.shuffle(dataset)
train, cv, test = np.split(dataset, [int(.6 * len(dataset)), int(.8 * len(dataset))])
X_train = train[:, :-1]
y_train = train[:, -1]
X_cv = cv[:, :-1]
y_cv = cv[:, -1]
X_test = test[:, :-1]
y_test = test[:, -1]

# feature scaling / mean normalization
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_cv = sc_X.transform(X_cv)
X_test = sc_X.transform(X_test)
'''

# fit random forest regressor
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100, criterion='mse')
reg.fit(X_train, y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=reg, X=X_train, y=y_train, cv=10)
print(accuracies.mean())
print(accuracies.std())

# Applyig Grid Search to find best model and best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[0.3,0.3125,0.325,0.3375,0.35], 'kernel' :['rbf'], 'gamma': [9.875,9.93,10,10.06,10.125]}
             ]
grid_search = GridSearchCV(estimator=reg,
                           param_grid = parameters,
                           scoring = 'f1',
                           cv=10,
                           n_jobs=-1)
grid_search = grid_search.fit(X_train,y_train)

print(grid_search.best_score_)
best_params = grid_search.best_params_

# visualize
X_grid = np.arange(min(X_train),max(X_train),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_grid,reg.predict(X_grid))
plt.show()

# predict new result with random forest regression
print(reg.predict(np.array([[6.5]])))