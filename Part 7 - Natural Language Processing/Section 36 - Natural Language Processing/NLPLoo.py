#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 21:23:37 2018

@author: cfzhou
"""


# import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)


# clean / stem texts
import nltk
directory = '/home/cfzhou/Projects/Udemy-ML-python/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing'
nltk.download('stopwords', download_dir = directory)
if directory not in nltk.data.path:
    nltk.data.path.append(directory)
    
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()

corpus = []
for ind in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][ind])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# Create Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
vocab = cv.vocabulary_
y = dataset.iloc[:, -1].values

print('cleaned!')

# split dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


# fit random forest model
from sklearn.ensemble import RandomForestClassifier
clas = RandomForestClassifier(n_estimators=10,
                              min_samples_leaf=1,
                              max_depth=None,
                              n_jobs=-1,
                              random_state=42)


# feature selection
from sklearn.feature_selection import RFECV

rfecv = RFECV(estimator=clas, 
              step=1, 
              cv=10,
              scoring='f1_micro',
              n_jobs=-1)
rfecv.fit(X_train, y_train)

relev_feat = rfecv.support_
# relev_feat = rfecv.ranking_ == 1

print("Optimal number of features : %d" % rfecv.n_features_)


# limit features
X_train = X_train[:,relev_feat]
X_test = X_test[:,relev_feat]
clas.fit(X_train, y_train)


# Applying Grid Search to find best model and best parameters

from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators':[10, 30, 100, 300 ,1000, 3000],
               'min_samples_leaf':[1, 3 ,10, 30, 100, 300, 1000, 3000, 10000],
               'max_depth':[10, 30, 100, 300, 1000, 3000, 10000],
               'random_state' :[42]
               }
             ]
grid_search = GridSearchCV(estimator=clas,
                           param_grid = parameters,
                           cv=10,
                           scoring='f1_micro',
                           verbose=0,
                           n_jobs=-1)
grid_search = grid_search.fit(X_train,y_train)

print(grid_search.best_score_)
best_params = grid_search.best_params_

clas = grid_search.best_estimator_


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=clas, 
                             X=X_train, 
                             y=y_train,
                             scoring='f1_micro',
                             cv=10)
print('\nk-foldscoring:',accuracies.mean())
print('k-foldscoring std:', accuracies.std())


# predict test set
y_pred = clas.predict(X_test)


# scoring
from sklearn import metrics
final_score = metrics.f1_score(y_test, y_pred, average='micro')















