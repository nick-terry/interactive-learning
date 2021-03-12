#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:47:09 2021

@author: nick
"""
import numpy as np
import matplotlib.pyplot as plt

import gOptElimination as elim
import linThompson as thomp
import linucb as ucb


def runExperiment(d,n,T,X,theta,seed):
    
    np.random.seed(seed)
    
    # Create bandits
    elimBandit = elim.GOptElimBandit(arms=X, horizon=T, theta=theta)
    
    elimBandit.play()
   
    # Run the other two methods for last value of t for G-Opt
    ucbBandit = ucb.LinUCBBandit(arms=X, horizon=elimBandit.t, lambd_reg=.2, theta=theta)
    thompBandit = thomp.LinThompsonBandit(arms=X, horizon=elimBandit.t, theta=theta)
     
    bandits = [ucbBandit,thompBandit,elimBandit]
    for bandit in bandits[:-1]:
        bandit.play()
    
    return bandits

def plotRegret(bandits,d,n,T):
    
    fig,ax = plt.subplots(1)
    for name,bandit in zip(('ucb','thompson','G-opt'),bandits):
        ax.plot(range(1,T+1),np.cumsum(bandit.regret[:T]),label=name)
        
    ax.legend()
    ax.set_xlabel('Round (t)')
    ax.set_ylabel('Cumulative Regret (R_t)')
    ax.set_title('d={}, n={}, T={}'.format(d,n,T))

if __name__=='__main__':
    
    seed = 1234
    np.random.seed(seed)
    
    d = 2
    n = 10000
    T = 100000
    
    theta = np.zeros((d,))
    theta[0] = 1
    
    # Draw points on unit sphere
    X = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=n)
    X = X/np.linalg.norm(X,ord=2,axis=1)[:,None]
    
    bandits = runExperiment(d,n,T,X,theta,seed)
    
    plotRegret(bandits, d, n, T)