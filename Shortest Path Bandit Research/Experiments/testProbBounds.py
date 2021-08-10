#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 09:09:26 2021

@author: nick
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat

d = 5

Z = np.eye(d)
xPrime = (np.cos(.01)*Z[0] + np.sin(.01)*Z[1])[None,:]
X = np.concatenate([Z,xPrime])
n = X.shape[0]
theta = Z[0]
theta = theta/np.linalg.norm(theta)
zStar = X[0]


maxNorm = np.max(np.linalg.norm(X-X[0],axis=1))

N = 100
diff = X-X[0]
sigmaHat = np.linalg.pinv(np.eye(d) + np.sum(N * X[:,None,:] * X[:,:,None],axis=0))
maxNorm = np.max(np.sum(diff @ sigmaHat * diff,axis=1))

yList = []
for x in X:
    y = x.T @ theta + np.random.normal(0,1,N)
    yList.append(y[:,None] * x[None,:])
    
thetaHat = np.squeeze(sigmaHat @ np.sum(np.concatenate(yList),axis=0))
    
def beta(delta,k):
    return np.sqrt(2*np.log((1+k)**(d/2)/delta)+1)

def upperBound(delta):
    _beta = beta(delta,1)
    return maxNorm * _beta * (1+np.sqrt(stat.chi2.ppf(1-delta,d)))

def gaussianProb(eps):
    return (d/(d+eps))**(-d/2)*np.exp(-eps/2)

def gaussianProb2(eps):
    return (d/eps**2)**(-d/2)*np.exp(-(eps**2-d)/2)

def gaussianUB(delta):
    return np.sqrt(2*d*np.log(2*d/delta))

# nDraws = int(1e7)

# Gaussian UB experiment
deltaR = np.arange(.05,.95,.01)
# X = np.random.multivariate_normal(np.zeros(d), np.eye(d), nDraws)
# xNorm = np.linalg.norm(X,ord=2,axis=1)
# # ub = gaussianUB(deltaR)
# pHat = np.mean(xNorm[:,None] > eps[None,:],axis=0)

# fig,ax = plt.subplots(1)
# ax.plot(deltaR,pHat)

# ax.set_xlabel('$\delta$')
# ax.set_ylabel('Empirical Prob')

# UB Experiment  
# thetaTilde = np.random.multivariate_normal(thetaHat, sigmaHat,nDraws)
# z_t = X[np.argmax(X @ thetaTilde.T,axis=0)]

# deltaHat = np.sum((zStar-z_t) * theta[None,:],axis=1)

# deltaR = np.arange(.01,1,.01)
# ubR = upperBound(deltaR)

# pHat = np.sum(deltaHat[:,None] > ubR[None,:],axis=0)/nDraws

# quantiles = np.quantile(deltaHat,deltaR)

# fig,ax = plt.subplots(1)
# ax.plot(deltaR,ubR,label='Upper Bound')
# ax.plot(1-deltaR,quantiles,label='Empirical Quantile')
# ax.set_xlabel('$\delta$')
# ax.set_ylabel('Bounds/Empirical Quantile')
# ax.legend()

# eps = np.arange(0,10,.1)

# X = np.random.multivariate_normal(np.zeros(d), np.eye(d), nDraws)
# xNorm = np.linalg.norm(X,ord=2,axis=1)

# pHat = np.mean(xNorm[:,None] > eps[None,:],axis=0)

# plt.plot(eps,p,label='upper bound')
# plt.plot(eps,pHat,label='empirical prob')
# plt.legend()

def getT(r,delta,d,Sigma):
    _beta = beta(delta,1)
    p = (r/2)**d /_beta**d *np.linalg.det(Sigma)
    return np.log((r/6)**d * delta)/np.log(1-p)

r = 1
tRange = getT(r,deltaR,d,np.eye(d))
fig,ax = plt.subplots(1)
ax.plot(deltaR,tRange)
ax.set_xticks(np.arange(0,1,.05))
ax.set_xlabel('$\delta$')
ax.set_ylabel('T')

dMax = 5
dRange = np.arange(1,dMax + 1,1)
tRange2 = getT(r,.05,dRange,np.eye(d))
fig,ax = plt.subplots(1)
ax.plot(dRange,tRange2,label='lower bound on T')
ax.plot(dRange,2**dRange,label='$2^d$')
ax.set_xticks(np.arange(0,dMax + 1,5))
ax.set_xlabel('d')
ax.set_ylabel('T')
ax.legend()