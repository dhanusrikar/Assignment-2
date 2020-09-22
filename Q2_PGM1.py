#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import random
#the number of samples to be generated
data_points = 1000000
#Original Hypothesis Values
theta = [3,1,2]


# In[4]:


#Function to generate Normal Distribution data samples
def normal_distribution(mu, sigma):
    #this is the list in which data will be added
    z = []
    for i in range(data_points):
        #two random seed values 
        t1 = random.random()
        t2 = random.random()
        #normally distributed random value
        x = np.cos(2*np.pi*t1)*np.sqrt(-2*np.log(t2))
        #normalise the random value according to given sigma and mu
        x = x * (sigma ** 0.5) + mu
        z.append(x)
    return np.array(z)


# In[6]:


x0 = np.ones([data_points])
x1 = normal_distribution(3, 4)
x2 = normal_distribution(-1,4)
#gauss noise 
epsilon = normal_distribution(0, 2)
#y values wihout noise
y = theta[0] * x0 + theta[1] * x1 + theta[2] * x2 
#y values with noise
y_gauss_noise = theta[0] * x0 + theta[1] * x1 + theta[2] * x2 + epsilon
"""
#to plot the generated data
sns.distplot(y, color = 'yellow')
sns.distplot(y_gauss_noise, color = 'blue')
sns.distplot(x1, color = 'green')
sns.distplot(x2, color = 'orange')
sns.distplot(epsilon, color = 'red')
"""

