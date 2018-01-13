#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 01:56:39 2018

@author: cfzhou
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("train.csv")
corr_mat = dataset.corr

dataset = dataset.iloc[:,1:].values

# encode categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
dataset[:, 0] = labelencoder.fit_transform(dataset[:, 0])

dataset = dataset.astype(float)

# spliting into training, CV, test sets
np.random.shuffle(dataset)
train, test = np.split(dataset, [int(.8 * len(dataset))])
X_train = train[:, 1:]
y_train = train[:, 0]
X_test = test[:, 1:]
y_test = test[:, 0]

# Fit Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
clas = GaussianNB()
clas.fit(X_train,y_train)


# add features
'''
from sklearn.preprocessing import PolynomialFeatures

poly_feat = PolynomialFeatures(degree = 1)
X_train = poly_feat.fit_transform(X_train)
X_cv = poly_feat.fit_transform(X_cv)
X_test = poly_feat.fit_transform(X_test)
'''

# Learning Curve
from sklearn.model_selection import learning_curve
plt.figure()
plt.title("Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Score")
train_sizes, train_scores, test_scores = learning_curve(
        clas, X_train, y_train, cv=10, n_jobs=-1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

plt.legend(loc="best")

# predict test set
y_pred = clas.predict(X_test)

# scoring

from sklearn import metrics
final_score = metrics.f1_score(y_test, y_pred, average='micro')


