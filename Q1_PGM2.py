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
    theta_0=[0,1]
    theta_1=[0,1]
    
    tolError=float(input("Enter the tolerable error = "))
    n=int(input("enter the maximum number of iterations = "))
    eta_d=float(input("Enter the seed value = "))
    
    iterations = 1
    temp=0

    
    while(abs(theta_0[-1]-theta_0[-2])>tolError and abs(theta_1[-1]-theta_1[-2])>tolError):
        eta=eta_d/(math.sqrt(iterations))
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

def MSE(X,Y,theta_0,theta_1):
    pred=theta_0+theta_1*X
    sqr=np.subtract(Y,pred)
    mean_square_error=np.square(sqr)
    mean_square_error=np.mean(mean_square_error)
    print("mean square error percentage= %0.8f"%(mean_square_error*100))
    

def plot_graph(X,Y,theta_0,theta_1):
    pred=theta_0+theta_1*X
    
    plt.plot(Y,"ro",label="actual y")
    plt.plot(pred,"bo",label="predicted y")
    #plt.plot(X,Y,'ro',label="actual x vs actual y",markersize=6)
    #plt.plot(X, pred, color='blue', marker='+',label="actual x vs predicted y", linestyle='dashed',linewidth=1, markersize=12,)
    plt.legend(loc="best")
    plt.xlabel("acidity")
    plt.ylabel("density")
    plt.title("actual V/S predicted(adaptive learning)")
    plt.show()
    
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#main function

print("\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX LINEAR REGRESSION XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

#predicting theta0 and theta1 values using train data
theta_0,theta_1=train_model(x_train,y_train)

#predicting mean square using test data
MSE(x_test,y_test,theta_0[-1],theta_1[-1])

#plotting graph
plot_graph(x_test,y_test,theta_0[-1],theta_1[-1])



