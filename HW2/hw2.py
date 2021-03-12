#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:13:08 2021

@author: nick
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# Problem 5.1
# Compute optimal sigma for minimax bound in 5.1
def b(sigma,k):

    b = np.sqrt(1+1/2/k**2/np.log(2/.05)/sigma**2)*\
        np.sqrt((2*np.log(1/.05) + np.log(2*k**2*np.log(2/.05)*sigma**2+1)))
    return b

sigmaList = []
kArr = np.linspace(1,100,1000)
for k in kArr:
    sigmaStar = scipy.optimize.minimize(b,1,args=(k,),
                                        constraints={'type':'ineq',
                                                      'fun':lambda x:x})
    sigmaList.append(sigmaStar.x)
    
plt.plot(1/kArr,sigmaList)
plt.xlabel('Absolute value of E[Z_i]')
plt.ylabel('Optimal value of sigma^2')

# Plot ratio of confidence bounds
def eq1(delta,sigmaSq,t):
    return np.sqrt(1+1/t/sigmaSq)*\
        np.sqrt((2*np.log(1/delta)+np.log(t*sigmaSq+1))/t)
        
def eq3(delta,t):
    return np.sqrt(2*np.log(2/delta)/t)

delta = .05
maxt = 10e6
tArr = np.linspace(1,int(maxt),int(maxt/100))
sigmaSqArr = 10**np.linspace(-6,0,7)

fig,ax = plt.subplots(1)
for sigmaSq in sigmaSqArr:
    ax.plot(tArr,eq1(delta,sigmaSq,tArr)/eq3(delta,tArr),
            label='sigmaSq={}'.format(sigmaSq))

plt.xlim(0,9e6)
plt.ylim(1,1.8)
plt.xlabel('t')
plt.ylabel('Ratio of confidence bounds')
plt.legend()

# Problem 5.2
def f(X,lambd,returnAll=False):
    
     # Create matrix A(lambda)
    A_lambd = np.sum(lambd[:,None,None] * (X[:,:,None]*X[:,None]),axis=0)
    
    # Compute quadratic form norm of each x_i w.r.t A(lambda)^{-1}
    AInv = np.linalg.inv(A_lambd)
    Xnorm = np.diag((X @ AInv) @ X.T)
    
    if returnAll:
        return np.max(Xnorm), AInv, Xnorm
    else:
        return np.max(Xnorm)

def df(X,lambd):
    """
    Compute the gradient of X w.r.t lambda

    Parameters
    ----------
    X : np array
        Design points. Each row of X is a vector x_i^T.
    lambd : float        

    Returns
    -------
    np array

    """
    
    _,AInv,Xnorm = f(X,lambd,returnAll=True)
    
    # Find x_i with largest norm and compute gradient 
    iMax = np.argmax(Xnorm)
    grad = - X[None,iMax,:] @ AInv @ (X[:,:,None] * X[:,None,:]) @ AInv @ X[None,iMax,:].T
    
    return grad.squeeze()

def randomSampling(X,N,step_size=1e-4):
    
    n = X.shape[0]

    nIter = 5000
    fArray = np.zeros((nIter,))
    lambd = np.ones((n,))/n
    fArray[0] = f(X,lambd)
    
    for step in range(nIter):
        
        lambd_tild = np.exp(np.log(lambd)-step_size*df(X,lambd))
        
        lambd = lambd_tild/np.sum(lambd_tild)
        
        fArray[step] = f(X,lambd)
        
    I = np.random.choice(np.linspace(0,n-1,n),N,replace=True)
    
    return I,fArray

# This one doesn't seem to work. Probably has to do with how I am computing lambda
def greedy(X,N):
    
    n = X.shape[0]
    d = X.shape[1]
    
    # Choose first 2d entries randomly
    _I = np.random.choice(range(n),size=2*d,replace=True)
    
    # Choose remaining entries greedily
    I = np.zeros((N,)).astype(int)
    I[:2*d] = _I.astype(int)
    
    fArray = np.zeros((N-2*d,))
    lambd = np.bincount(I[I>0],minlength=n)/np.sum(I>0)
    fArray[0] = f(X,lambd)
    
    # Sum the outer products of design points corresponding to first 2*d components of I
    xMat = np.sum(X[I[:2*d],:,None] * X[I[:2*d],None,:],axis=0)
    
    for t in range(2*d+1,N):
            
        # Add outer product of each design point (as different matrix)
        testMat = (X[:,:,None]*X[:,None]) + xMat[None,:,:]
        
        # Invert
        testMatInv = np.linalg.inv(testMat)        
        
        # Compute quadratic form norm of each x_i w.r.t A(lambda)^{-1}
        # repX = np.repeat(X[:,:,None],n,axis=-1)
        repX = np.repeat(X[None,:,:,],n,axis=0)
        # repInv = np.repeat(testMatInv[:,None,:,:],n,axis=1)
        
        # Einstein summation. k is x_i^t design point, l is index of extra additive term in test matrix
        # quad = np.einsum('kji,lji->kli', repX, testMatInv)
        quad = np.transpose(repX @ testMatInv,(1,0,2))
        
        # Finish computing quadratic form
        quad = np.einsum('kli,kli->kl', quad, np.transpose(repX,(1,0,2)))
        
        # Take max over argument of norm, then take argmin over all design points
        # Update I vector
        I[t] = np.argmin(np.max(quad,axis=0))
        
        # Update xMat
        xMat += X[I[t],:,None] * X[I[t],None,:]
        
        # Compute empirical distribution and objective function value
        lambd = np.bincount(I[I>0],minlength=n)/np.sum(I>0)
        fArray[t-(2*d+1)+1] = f(X,lambd)
        
    return I,fArray
        
def frankWolfe(X,N):
    
    n = X.shape[0]
    d = X.shape[1]
    
    # Choose first 2d entries randomly
    _I = np.random.choice(range(n),size=2*d,replace=True)
    
    # Choose remaining entries greedily
    I = np.zeros((N,)).astype(int)
    I[:2*d] = _I
    
    lambd = np.bincount(_I,minlength=n)/2/d
    fArray = np.zeros((N-2*d,))
    fArray[0] = f(X,lambd)
    
    for t in range(2*d+1,N):
        
        # Find j that minimizes gradient
        grad = df(X,lambd)
        I[t] = np.argmin(grad)
        
        # Update lambd
        eta_t = 2/(t+1)
        lambd = (1-eta_t) * lambd
        lambd[I[t]] += eta_t
        
        fArray[t-(2*d+1)+1] = f(X,lambd)
        
    return I,fArray

def runExperiment(a,N,d):
    
    sigmaSq = np.linspace(1,d,d)**(-a)
    
    # Randomly generate X from N(0,sigmaSq)
    X = np.random.multivariate_normal(np.zeros((d,)), np.diag(sigmaSq), size=(10 + 2**10,))
    
    nArray = (2**np.linspace(1,10,10) + 10).astype(int)
    methodNArr = np.zeros((3,nArray.shape[0]))
    for i,n in enumerate(nArray):   
        
        methodNArr[0,i] = randomSampling(X[:n],N)[1][-1]
        methodNArr[1,i] = greedy(X[:n],N)[1][-1]
        methodNArr[2,i] = frankWolfe(X[:n],N)[1][-1]
    
    return methodNArr

def runExperiments():
    
    N = 1000
    d = 10
    nArray = (2**np.linspace(1,10,10) + 10).astype(int)
    
    resultsList = []
    for a in [0,.5,1,2]:
        
        results = runExperiment(a, N, d)
        resultsList.append(results)
    
    return resultsList
    
def plotResults(results):
    
    nArray = (2**np.linspace(1,10,10) + 10).astype(int)
    
    for a,result in zip([0,.5,1,2],results):
        
        fig,ax = plt.subplots(1)
        
        for i,methodName in enumerate(('Random Sampling','Greedy','Frank-Wolfe')):
        
            ax.plot(nArray,result[i],label=methodName)
            
            ax.set_title('a={}'.format(a))
            ax.set_ylim((8,20))
            plt.xlabel('n')
            plt.ylabel('f(\lambda)')
            plt.legend()

resultsList = runExperiments()
plotResults(resultsList)