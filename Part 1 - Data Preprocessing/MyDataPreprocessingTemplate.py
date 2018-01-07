#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 20:52:09 2018

@author: cfzhou
"""

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
