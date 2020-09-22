#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import random
data_points = 1000000
theta = [3,1,2]


# In[2]:


def normal_distribution(mu, sigma):
    z = []
    for i in range(data_points):
        t1 = random.random()
        t2 = random.random()
        x = np.cos(2*np.pi*t1)*np.sqrt(-2*np.log(t2))
        x = x * (sigma ** 0.5) + mu
        z.append(x)
    return np.array(z)


# In[5]:


#initialise Theta values as 0,1
theta1 = [0,1]
theta2 = [0,1]
theta0 = [0,1]
batch_size = int(input('Enter batch size:'))
eta = 0.001
temp1 = temp2 = temp3 = 0


# In[4]:


x0 = np.ones([data_points])
x1 = normal_distribution(3, 4)
x2 = normal_distribution(-1,4)
#gauss noise 
epsilon = normal_distribution(0, 2)
#y values wihout noise
y = theta[0] * x0 + theta[1] * x1 + theta[2] * x2 
#y values with noise
y_gauss_noise = theta[0] * x0 + theta[1] * x1 + theta[2] * x2 + epsilon


# In[ ]:


#The following 3 functions calculate the partial derivaties for gradient
def djdt0(X1,X2,Y,n, theta0, theta1, theta2):
    ans = 0.0
    for i in range(n):
        ans = ans + (theta0 + theta1*X1[i] + theta2*X2[i] - Y[i])
    return ans/n

def djdt1(X1,X2,Y,n, theta0, theta1, theta2):
    ans = 0.0
    for i in range(n):
       ans = ans + (theta0 + theta1*X1[i] + theta2*X2[i] - Y[i]) * (X1[i])
    return ans/n

def djdt2(X1,X2,Y,n, theta0, theta1, theta2):
    ans = 0.0
    for i in range(n):
       ans = ans + (theta0 + theta1*X1[i] + theta2*X2[i] - Y[i]) * (X2[i])
    return ans/n


# In[ ]:


#tcalculates the cost of the hypothesis
def cost_function(X1, X2, Y, theta0, theta1, theta2, noise):
    j = 0
    for i in range(len(X1)):
        temp = (Y[i] - (theta0 + (theta1 * X1[i]) + (theta2 * X2[i]))) ** 2
        j = j +temp
    #1/(2*len(y)) = 1 / 2m
    return j/(2*len(Y))


# In[ ]:


#Shuffle the data
def shuffle_Data(x1,x2,y_gauss_noise,epsilon):
    mapIndexPosition = list(zip(x1, x2,epsilon,y_gauss_noise))
    random.shuffle(mapIndexPosition)
    x1_s,x2_s,noise_s,y_s = zip(*mapIndexPosition)
    x1 = x1_s
    x2 = x2_s
    epsilon = noise_s
    y_gauss_noise = y_s
    return x1, x2, y_gauss_noise, epsilon


# In[ ]:


#a and b are calculated to find the initial error when theta is 0 and 1
a = cost_function(x1,x2,y_gauss_noise,theta0[-1], theta1[-1], theta2[-1], epsilon)
b = cost_function(x1,x2,y_gauss_noise,theta0[-2], theta1[-2], theta2[-2], epsilon)
x1, x2, y_gauss_noise, epsilon = shuffle_Data(x1,x2,y_gauss_noise,epsilon)
tol_error = 0.00001
error = abs(a - b)
#q = 10
while(error > tol_error):
    #print('yay')
    initial = 0
    #iterate over the data in multiples of batch size
    for i in range(1, int(len(x1)//batch_size)):
        #take the data of specific batch 
        noise = epsilon[initial:i*batch_size]
        X1 = x1[initial:i*batch_size]
        X2 = x2[initial:i*batch_size]
        Y = y_gauss_noise[initial:i*batch_size]
        #calculate THETA(t+1) = THETA(t) - ((eta)*(gradientDescent))
        temp1 = theta0[-1] - (eta * djdt0(X1, X2, Y,batch_size,theta0[-1], theta1[-1], theta2[-1]))
        temp2 = theta1[-1] - (eta * djdt1(X1, X2, Y,batch_size,theta0[-1], theta1[-1], theta2[-1]))
        temp3 = theta2[-1] - (eta * djdt2(X1, X2, Y,batch_size,theta0[-1], theta1[-1], theta2[-1]))
        theta0.append(temp1)
        theta1.append(temp2)
        theta2.append(temp3)
        #increment to next batch
        initial = i*batch_size
        #print(theta0[-1], theta1[-1], theta2[-1])
    a = cost_function(X1,X2,Y,theta0[-1], theta1[-1], theta2[-1], noise)
    b = cost_function(X1,X2,Y,theta0[-2], theta1[-2], theta2[-2], noise)
    error = abs(a - b)
    final_theta = [theta0[-1], theta1[-1], theta2[-1]]
    x1, x2, y_gauss_noise, epsilon = shuffle_Data(x1,x2,y_gauss_noise,epsilon)
    #print(a,b,error)
    #q = q - 1    
print('theta0: ',theta0[-1], 'theta1: ', theta1[-1], 'theta2: ', theta2[-1])


# In[ ]:




