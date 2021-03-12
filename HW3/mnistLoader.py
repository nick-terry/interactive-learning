#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:32:03 2021

@author: nick
"""

import numpy as np
from mnist import MNIST

def loadMNIST():
    
    mndata = MNIST('./data/')
    X_train, y_train = map(np.array, mndata.load_training())
    X_test, y_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    
    return X_train,y_train,X_test,y_test

def getEig(X):
    
    # compute mean of training set
    mu = np.mean(X,axis=0)
    
    # compute sample covar matrix
    diff = X - mu
    sigma = diff.T @ diff / X.shape[0]
    
    Lambda, V = np.linalg.eig(sigma)
    
    return Lambda, V

def project(X,basis):
    
    # project X onto basis w/ inner products
    proj = X @ basis
    
    # compute reconstruction as linear comb of eigenvectors
    # reconstr = proj @ basis.T
    
    return proj

def getRepresentation(X,d,V=None,retV=False):
    # Get a d-dimensional representation of X using PCA
    
    if V is None:
        Lambda, V = getEig(X)
        
    basis = np.real(V[:,:d])
    representation = project(X,basis)
    
    if retV:
        return representation,V
    else:
        return representation
    
def rescale(X):
    # Rescale the rows of X to have unit L2 norm
    
    return X/np.linalg.norm(X,ord=2,axis=1)[:,None]
    
