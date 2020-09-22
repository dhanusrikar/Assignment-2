#!/usr/bin/env python
# coding: utf-8

# In[5]:


from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

theta0 = np.array(pd.read_csv('batch_1.csv',usecols = ['theta0']))
theta1 = np.array(pd.read_csv('batch_1.csv',usecols = ['theta1']))
theta2 = np.array(pd.read_csv('batch_1.csv',usecols = ['theta2']))
print(len(theta0))


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(theta1,theta2,theta0)
plt.ylabel('Theta1')
plt.xlabel('Theta2')
plt.show()


# In[ ]:




