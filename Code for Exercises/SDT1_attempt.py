# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 14:06:28 2016

@author: will
"""
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

"""
Question 1
"""
x = np.linspace(-4, 4, 100)
pdf = stats.norm.pdf(x)        # distribution
d_prime=[0.25,0.5,1.0,3.0,7.0] # sensitivity values
lamda=np.zeros(5)
for i in np.arange(0,5):
    lamda=np.linspace(0,d_prime[i],5)
    plt.subplot(2,3,i+1)
    plt.plot(x+d_prime[i],pdf) # noise pdf
    plt.plot(x,pdf)            # signal pdf
    plt.plot([lamda,lamda+0.01],[0,0.4],'k') # critereon
    
# The above took me just over half an hour
    
    
"""
Question 2
"""
# Hits vs False Alarms with the same d' values
lamda=np.zeros(5)
for i in np.arange(0,5):
    lamda=np.linspace(0,d_prime[i],5)
    hits= 1-stats.norm.cdf(lamda-d_prime[i])
    FA  = 1-stats.norm.cdf(lamda)
    plt.subplot(2,3,6)
    plt.plot(FA,hits,'o-')
    plt.xlabel("False Alarms")
    plt.ylabel("Hits")
    plt.xlim([0,1])
    plt.ylim([0,1])

# This took me 10 minutes given the equations for hits and FA