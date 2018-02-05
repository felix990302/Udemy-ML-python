#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 23:04:31 2018

@author: cfzhou
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# UCB Algorithm
import math

N = 10000
d = 10

ad_selected = []
num_select = [0]*d 
sums_rewards = [0]*d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_UB = 0;
    for i in range(0, d):
        if(num_select[i]>0):
            avg_reward = sums_rewards[i] / num_select[i]
            delta = math.sqrt(3/2 * math.log(n+1) / num_select[i])
            UCB = avg_reward + delta
        else: 
            UCB = math.inf
        if(UCB>max_UB):
            max_UB = UCB
            ad = i
    ad_selected.append(ad)
    num_select[ad] += 1
    reward = dataset.values[n,ad]
    sums_rewards[ad] += reward
    total_reward += reward
    
# Visualize

plt.hist(ad_selected)

    
    
    
    
    