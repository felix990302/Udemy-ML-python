#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 15:42:46 2018

@author: cfzhou
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("cluster.csv").iloc[:,1:]

dataset = dataset.values

# missing data
'''
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
dataset[:, 1:3] = imputer.fit_transform(dataset[:, 1:3])
'''

# encode categorical data
'''
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
dataset[:, 0] = labelencoder.fit_transform(dataset[:, 0])
'''
dataset = dataset.astype(float)


# feature scaling / mean normalization
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train[:,1:] = sc.fit_transform(train[:,1:])
test[:,1:] = sc.transform(test[:,1:])
'''

# Find optimal cluster
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    clus = KMeans(n_clusters=i, 
                  init='k-means++', 
                  max_iter=300, 
                  n_init=10,
                  random_state=0,
                  n_jobs=-1)
    clus.fit(dataset)
    
    wcss.append(clus.inertia_)

plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.show()

# K-means w/ optimal number of centroids
clus = KMeans(n_clusters=5, 
              init='k-means++', 
              max_iter=300, 
              n_init=10,
              random_state=0,
              n_jobs=-1)
y_kmeans = clus.fit_predict(dataset)


# visualize clusters

for i in range(0,5):
    plt.scatter(dataset[y_kmeans==i, 0], dataset[y_kmeans==i, 1], label = 'Cluster %d'%i)

plt.scatter(clus.cluster_centers_[:,0], clus.cluster_centers_[:,1], s = 300, label='Centroids')




