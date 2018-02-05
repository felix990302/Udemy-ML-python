#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 15:25:16 2018

@author: cfzhou
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("cluster.csv").iloc[:,1:]

dataset = dataset.values

X = dataset.astype(float)

# feature scaling / mean normalization
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(dataset)
'''


# Find optimal cluster
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('X')
plt.xlabel('y')
plt.show()

# fitting to dataset
from sklearn.cluster import AgglomerativeClustering
clus = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_clus = clus.fit_predict(X)


# visualize clusters
'''
for i in range(0,5):
    plt.scatter(dataset[y_kmeans==i, 0], dataset[y_kmeans==i, 1], label = 'Cluster %d'%i)

plt.scatter(clus.cluster_centers_[:,0], clus.cluster_centers_[:,1], s = 300, label='Centroids')
'''



