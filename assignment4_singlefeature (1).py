#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#feature 1- Economy
#feature 2- Family
#feature 3- Health
#feature 4- Freedom
#feature 5- Trust
#feature 6- Generosity
#feature 7- Dystopia Residual

import pandas as pd
import numpy as np
import math
import csv
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def initalize_weights(size_layer, size_next_layer):
    np.random.seed(100)
    epsilon = np.sqrt(2.0 / (size_layer * size_next_layer) )
    w = epsilon * (np.random.randn(size_next_layer, size_layer))
    return w.transpose()

def normalize_data(data):
    data_new = data.copy()
    max_num = data.max()
    min_num = data.min()
    data_new = (data - min_num)/(max_num - min_num)
    return data_new

def load_file():
    hp1=pd.read_csv('2017.csv')
    hp2=pd.read_csv('2016.csv')
    hp3=pd.read_csv('2015.csv')
    col_2015=['Region','Standard Error']
    hp3_mod=hp3.drop(col_2015,axis=1)
    col_2016=['Region','Lower Confidence Interval','Upper Confidence Interval']
    hp2_mod=hp2.drop(col_2016,axis=1)
    col_2017=['Whisker.high','Whisker.low']
    hp1_mod=hp1.drop(col_2017,axis=1)
    final=[hp3_mod,hp2_mod,hp3_mod]
    finaltable=pd.concat(final)
    Y = finaltable.iloc[:,2].values
    X = finaltable.iloc[:,3:].values
    X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=1)
    X_train, X_val, Y_train, Y_val =train_test_split(X_train, Y_train, test_size=0.25, random_state=1)
    return X_train,X_test,X_val,Y_train,Y_test,Y_val
        
epochs = 1000
loss = np.zeros([epochs,1])
alpha=0.1
X_train,X_test,X_val,Y_train,Y_test,Y_val= load_file()
tic = time.time()

def compute_cost(X,Y,i,j,k):
    errorfeature=[]
    XX=np.copy(X)
    Y=Y.reshape(Y.shape[0],1)
    Y1=normalize_data(Y)
    for k1 in range(7):
        XX[:,k1]=normalize_data(X[:,k1])
    n_examples = X.shape[0]
    X2=XX[:,k]
    X2=X2.reshape(X2.shape[0],1)
    w1 = initalize_weights(2, i)
    w2 = initalize_weights(i, j)
    w3 = initalize_weights(j,Y1.shape[1])
    X1=np.concatenate((np.ones((X2.shape[0],1)),X2),axis=1)
    for ix in range(epochs):  
        a1 = X1
        z2 = a1.dot(w1)
        a2 = np.tanh(z2)
        z3 = a2.dot(w2)
        a3 = np.tanh(z3)
        z4 = a3.dot(w3)
        a4 = z4
        Y_hat = a4
        loss[ix] = np.sqrt((0.5) * np.square(Y_hat - Y1).mean())
        d4 = Y_hat - Y1
        grad3 = a3.T.dot(d4)/n_examples
        d3 = d4.dot(w3.T)
        d3 = d3*(1-a3*a3)
        grad2 = a2.T.dot(d3)/n_examples
        d2 = d3.dot(w2.T)
        d2 = d2*(1-a2*a2) 
        grad1 = a1.T.dot(d2)/n_examples
        w1 = w1 - alpha*grad1
        w2 = w2 - alpha*grad2
        w3 = w3 - alpha*grad3
    return(loss.min())
featurevec=[]
for i in range(1,100,5):
    for k in range(7):
        pl=[]
        nl=[]
        a = compute_cost(X_train,Y_train,i,i,k)
        b = compute_cost(X_val,Y_val,i,i,k)
        c = compute_cost(X_test,Y_test,i,i,k)
        pl.append(b)
        nl.append([i,i])
        a = compute_cost(X_train,Y_train,i,100-i,k)
        b = compute_cost(X_val,Y_val,i,100-i,k)
        c = compute_cost(X_test,Y_test,i,100-i,k)
        pl.append(b)
        nl.append([i,100-i])
        a = compute_cost(X_train,Y_train,100-i,i,k)
        b = compute_cost(X_val,Y_val,100-i,i,k)
        c = compute_cost(X_test,Y_test,100-i,i,k)
        pl.append(b)
        nl.append([100-i,i])
        index=pl.index(min(pl))
        featurevec.append([k+1,min(pl),nl[index]])
for j2 in range(7):
    features=[]
    error=[]
    neurons=[]
    for i2 in range(j2,len(featurevec),7):
        features.append(featurevec[i2][0])
        error.append(featurevec[i2][1])
        neurons.append(featurevec[i2][2])
    indexpos=error.index(min(error))
    print('for features',features[0],'lowest crossvalidation error is',error[indexpos],' with number of neurons in 1st and 2nd hidden layers are',neurons[indexpos])


# In[ ]:




