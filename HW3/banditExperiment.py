#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:47:09 2021

@author: nick
"""
import numpy as np
import matplotlib.pyplot as plt

import mnistLoader as mn
import linThompson as thomp
import linucb as ucb
import etc
import etcBias as etcb
import followLeader as ftl

def runExperiment(T,X,tau1,tau2,gamma,contextPool,labelPool,seed):
    # tau1 is for ETC
    # tau2 is for FTL
    
    np.random.seed(seed)
    
    # Create bandits
    etcBandit = etc.ETCBandit(arms=X, horizon=T, tau=tau1, 
                              contextPool=contextPool, labelPool=labelPool)
    etcbBandit = etcb.ETCBiasBandit(arms=X, horizon=T, tau=tau1, 
                              contextPool=contextPool, labelPool=labelPool)
    ftlBandit = ftl.FTLBandit(arms=X, horizon=T, tau=tau2, 
                              contextPool=contextPool, labelPool=labelPool)
    ucbBandit = ucb.LinUCBBandit(arms=X, horizon=T, lambd_reg=gamma, 
                                 contextPool=contextPool, labelPool=labelPool)
    thompBandit = thomp.LinThompsonBandit(arms=X, horizon=T, lambd_reg=.5, 
                                 contextPool=contextPool, labelPool=labelPool)
     
    bandits = [etcBandit,etcbBandit,ftlBandit,ucbBandit,thompBandit]
    for bandit in bandits:
        bandit.play()
    
    return bandits

def plotRegret(bandits,tau1,tau2,gamma):
    
    fig,ax = plt.subplots(1)
    for name,bandit in zip(('ETC_MTW','ETC_MTB','FTL','UCB','Thomp'),bandits):
        ax.plot(range(1,T+1),np.cumsum(bandit.regret[:T]),label=name)
        
    ax.legend()
    ax.set_xlabel('Round (t)')
    ax.set_ylabel('Cumulative Regret (R_t)')
    ax.set_title('tau_1={}, tau_2={}, gamma={}'.format(tau1,tau2,gamma))

if __name__=='__main__':
    
    seed = 1234
    np.random.seed(seed)
    
    np.random.seed(1234)
    
    T = 50000
    
    X_train,y_train,X_test,y_test = mn.loadMNIST()
    contextPool = mn.rescale(mn.getRepresentation(X_train,d=16))
    labelPool = y_train
    X = np.unique(y_train)
    
    bigPhiSize = 1
    tau1 = np.ceil((X.shape[0]*T**2 * np.log(2*bigPhiSize))**(1/3))
    tau2 = tau1/2
    gamma = 1
    
    bandits = runExperiment(T,X,tau1,tau2,gamma,contextPool,labelPool,seed)
    newThompBandit = thompBandit = thomp.LinThompsonBandit(arms=X, horizon=T, lambd_reg=.5, 
                                 contextPool=contextPool, labelPool=labelPool)
    newThompBandit.play()
        
    plotRegret(bandits, tau1, tau2, gamma)