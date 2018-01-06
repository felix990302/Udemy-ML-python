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
dataset = pd.read_csv("50_Startups.csv")

# missing data
'''
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
dataset[:, 1:3] = imputer.fit_transform(dataset[:, 1:3])
'''

# encode categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
dataset.iloc[:, 3] = labelencoder.fit_transform(dataset.iloc[:, 3])

onehotencoder = OneHotEncoder(categorical_features=[3])
dataset = onehotencoder.fit_transform(dataset).toarray()

# removing summy variables
dataset = dataset[:,1:]

# spliting into training, CV, test sets
dataset = pd.DataFrame(dataset)

X = np.append(arr=np.ones((len(dataset),1)), values=dataset.iloc[:,:-1].values, axis=1)
y = dataset.iloc[:,-1].values

train, test = np.split(dataset.sample(
    frac=1), [int(.8 * len(dataset))])
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values
X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

# fitting parameters
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize=True)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# cost fens
# cost = np.sum(np.square(np.subtract(y_pred, y_train))) / (2*len(y_train))

# optimize using backward elimination
import statsmodels.formula.api as sm
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

