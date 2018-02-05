#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 23:52:35 2018

@author: cfzhou
"""

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
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None).iloc[:,:]

dataset=dataset.values

transactions = []
for i in range(0,len(dataset)):
        transactions.append([str(dataset[i,j]) for j in range(0,len(dataset[0,:]))])


# Training Apriori
from apyori import apriori
rules = apriori(transactions, 
                min_support=0.003, # bought once per day at least
                min_confidence=0.2,
                min_lift=3,
                min_length = 2
                )

results = list(rules)






