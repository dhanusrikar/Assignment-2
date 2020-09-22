# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:44:03 2020

@author: Avinash
"""

"""question=Implement batch gradient descent method for optimizing J(θ). Choose an
appropriate learning rate and the stopping criteria (as a function of the
change in the value of J(θ)). You can initialize the parameters as θ = V(0) (the
vector of all zeros). Do not forget to include the intercept term. Report your
learning rate, stopping criteria and the final set of parameters obtained by
your algorithm."""

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

def error(X,Y,theta_0,theta_1,hypo):
    totalError = 0
    for i in range(len(X)):
        totalError += (Y[i] - (theta_0*X[i] + theta_1)) ** 2
    return totalError/ float(len(hypo))
    

def plot_graph(X,Y,theta_0,theta_1):
    for i in range(len(theta_0)):
        hypo = theta_0[i] + theta_1[i] * X
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ms = np.linspace(theta_1[i]+0.5, theta_1[i]-0.5, 20)
        bs = np.linspace(theta_0[i]+0.5, theta_0[i]-0.5, 20)
        M,B = np.meshgrid(ms, bs)
        zs = np.array([error(X,Y,mp,bp,hypo) for mp, bp in zip(np.ravel(M), np.ravel(B))])
        Z = zs.reshape(M.shape)
        ax.plot_surface(M, B, Z, rstride=1, cstride=1, color='r', alpha=0.5,cmap = 'autumn')
    
        ax.set_xlabel('m')
        ax.set_ylabel('b')
        ax.set_zlabel('error')
        ax.view_init(30,20)
        plt.show()
        plt.pause(0.2)
    
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#main function

print("\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX LINEAR REGRESSION XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

#predicting theta0 and theta1 values using train data
theta_0,theta_1=train_model(x_train,y_train)



#plotting graph
plot_graph(x_test,y_test,theta_0,theta_1)




