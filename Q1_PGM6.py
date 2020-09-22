# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 20:33:11 2020

@author: Avinash
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import sleep 
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#data acquisition

#reading the input csv files

X=pd.read_csv("linearX.csv",header=None)


Y=pd.read_csv("linearY.csv",header=None)

#converting the datasets into arrays
X=np.array(X)
Y=np.array(Y)

#normalising the data
Mean=np.mean(X)
X1=(X-Mean)/np.var(X)
X=X1

#splitting the data into train_set and test_set
x_train, x_test , y_train, y_test = train_test_split(X,Y,test_size = 0.20,random_state = 42)

#print("normalised acidity values = ",X)
#print("density values = ",Y)

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#calculation part

def djdt0(theta_0,theta_1,m,X,Y):
    ans=0
    for i in range(m):
        ans=ans+(theta_0+(theta_1*X[i])-Y[i])
    return float(ans[0])

def djdt1(theta_0,theta_1,m,X,Y):
    ans=0
    for i in range(m):
        ans=ans+((theta_0+theta_1*X[i]-Y[i])*(X[i]))
    return float(ans[0])

def train_model(X,Y):
    m=len(X)

    #hyperparamters
    #initial theta assumptions
    theta_0=[0,0]
    theta_1=[0,0]
    
    tolError=float(input("Enter the tolerable error = "))
    n=int(input("enter the maximum number of iterations = "))
    eta=float(input("Enter the learning rate = "))
    
    iterations = 1
    temp=0
    temp1=theta_0[-1]-(eta*djdt0(theta_0[-1],theta_1[-1],m,X,Y))
    temp2=theta_1[-1]-(eta*djdt1(theta_0[-1],theta_1[-1],m,X,Y))
    theta_0.append(temp1)
    theta_1.append(temp2)
    iterations=iterations+1
    
    while(abs(theta_0[-1]-theta_0[-2])>tolError and abs(theta_1[-1]-theta_1[-2])>tolError):
        
        temp1=theta_0[-1]-(eta*djdt0(theta_0[-1],theta_1[-1],m,X,Y))
        temp2=theta_1[-1]-(eta*djdt1(theta_0[-1],theta_1[-1],m,X,Y))
        theta_0.append(temp1)
        theta_1.append(temp2)
        iterations=iterations+1
        
        if(iterations>n):
            temp=1
            break
        
    if temp==0:
        print("\ntotal no of iterations = ",iterations)
        print("the value of theta0 is = ",theta_0[-1])
        print("the value of theta1 is = ",theta_1[-1])

    else :
        print("OSCILLATING")
    
    return theta_0,theta_1

def error(X,Y,theta_0,theta_1):
    totalError = 0
    for i in range(len(X)):
        totalError += (Y[i] - (theta_0*X[i] + theta_1)) ** 2
    return totalError/ float(len(X))
    

def plot_graph(X,Y,theta_0,theta_1):
    for i in range (len(theta_0)):
        fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #values for theta1 axes of 3d plot
        ms = np.linspace(theta_1[i]+0.05, theta_1[i]-0.05, 10)
        #values for theta0 axes of 3d plot
        bs = np.linspace(theta_0[i]+0.05, theta_0[i]-0.05, 10)
        #creates a co-ordinate system of theta0 and theta1
        M,B = np.meshgrid(ms, bs)
        #creates the z-surface by calculating the error function
        zs = np.array([error(X,Y,mp,bp) for mp, bp in zip(np.ravel(M), np.ravel(B))])
        #converting the zs array from 1d to 2d array
        Z = zs.reshape(M.shape)
        #ploting the mesh
        cp=plt.contour(M,B,Z, 50, cmap = "Blues_r")
        plt.clabel(cp,inline=1,fontsize = 8)
        plt.xlabel('Theta 0')
        plt.ylabel('Theta 1')
        plt.title('Contours of the cost function')
        plt.show()
        sleep(0.2)
    
    
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#main function

print("\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX LINEAR REGRESSION XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

#predicting theta0 and theta1 values using train data
theta_0,theta_1=train_model(x_train,y_train)



#plotting graph
plot_graph(x_test,y_test,theta_0,theta_1)
