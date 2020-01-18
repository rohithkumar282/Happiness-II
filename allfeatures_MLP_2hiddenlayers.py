#!/usr/bin/env python
# coding: utf-8

# In[64]:


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
X_train,X_test,X_val,Y_train,Y_test,Y_val = load_file()
tic = time.time()

def compute_cost(X,Y,i,j):
    XX=np.copy(X)
    Y=Y.reshape(Y.shape[0],1)
    Y1=normalize_data(Y)
    for k in range(7):
        XX[:,k]=normalize_data(X[:,k])
    n_examples = X.shape[0]
    w1 = initalize_weights(XX.shape[1]+1, XX.shape[1]+i)
    w2 = initalize_weights(XX.shape[1]+i, XX.shape[1]+j)
    w3 = initalize_weights(XX.shape[1]+j,Y1.shape[1])
    X1=np.concatenate((np.ones((XX.shape[0],1)),XX),axis=1)
    for ix in range(epochs):
        n_examples = X1.shape[0]   
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
    return (loss.min(),XX.shape[1]+i,XX.shape[1]+j)

pl=[]
kl=[]
ll=[]
nl1=[]
nl2=[]
nl3=[]

for i in range(1,100,10):
    a1,a2,a3 = compute_cost(X_train,Y_train,i,i)
    b1,b2,b3 = compute_cost(X_val,Y_val,i,i)
    c1,c2,c3 = compute_cost(X_test,Y_test,i,i)
    pl.append(b1)
    nl1.append([b2,b3])
    a1,a2,a3 = compute_cost(X_train,Y_train,i,100-i)
    b1,b2,b3 = compute_cost(X_val,Y_val,i,100-i)
    c1,c2,c3 = compute_cost(X_test,Y_test,i,100-i)
    kl.append(b1)
    nl2.append([b2,b3])
    a1,a2,a3 = compute_cost(X_train,Y_train,100-i,i)
    b1,b2,b3 = compute_cost(X_val,Y_val,100-i,i)
    c1,c2,c3 = compute_cost(X_test,Y_test,100-i,i)
    ll.append(b1)
    nl3.append([b2,b3])
listmin=[]
listpos=[]
listcverror=[]
listcverror.append(pl)
listcverror.append(kl)
listcverror.append(ll)
print(' Cross validation errors are:')
print(listcverror)
listmin.append(min(ll))
listmin.append(min(kl))
listmin.append(min(pl))
listpos.append(nl3[ll.index(min(ll))])
listpos.append(nl2[kl.index(min(kl))])
listpos.append(nl1[pl.index(min(pl))])
ind=listmin.index(min(listmin))
q1=listpos[ind]

print(' The best model with four layers including input and output layer chosen for all features having lower Cross validation error',min(listmin),'with ',q1[0],'neurons in first hidden layer and ',q1[1],'neurons in the second hidden layer')
    

