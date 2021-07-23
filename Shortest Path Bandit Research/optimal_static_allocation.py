#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:05:21 2021

@author: nick
"""

import numpy as np
import torch as t
from scipy.optimize import linprog
import matplotlib.pyplot as plt

class StaticAllocation:
    
    def __init__(self,arms,Z,initAlloc,theta,delta=.05,eta=1e-5,sigma=1):
        """
        Implementation of a linear transductive bandit algorithm based on iteratively 
        refining a deterministic sampling allocation

        Parameters
        ----------
        arms : TYPE
            DESCRIPTION.
        Z : np array
            Choices (vectors) from which we are trying to maximize z^T \theta
        initAlloc : np array
            Initial allocation over the arms
        horizon : TYPE
            DESCRIPTION.
        lambd_reg : TYPE
            DESCRIPTION.
        theta : TYPE
            DESCRIPTION.
        N : int
            Number of samples to draw from allocation at each stage.
        eta : float
            Step size for optimizing allocation
        sigma : float, optional
            Variance of Gaussian noise added to rewards. The default is 1.

        Returns
        -------
        None.

        """
        
        self.arms = arms
        self.Z = Z
        
        # True theta used to generate rewards
        self.theta = theta
        self.zStar = np.argmax(theta)
        self._eta = eta
        self.sigma = sigma
        self.delta = delta
        
        self.initAlloc = initAlloc
        
        # Dimension of the action space
        self.d = self.arms.shape[1]
        # Number of possible actions (arms)
        self.n = self.arms.shape[0]
        
        self.sampleComplexity = 0
        
        self.zHat = None
        
    def pull(self,arms):
        """
        Pull arms and generate random rewards

        Parameters
        ----------
        arm : int
            Index of the arm to pull

        Returns
        -------
        outcome : float
            The random reward.

        """
        
        action = self.arms[arms]
        outcome = np.dot(action,self.theta) + np.random.normal(0,self.sigma**2,size=action.shape[0])
        
        return outcome
    
    def estimate(self,arms,rewards):
        """
        Compute the regularized least squares estimator for theta. This should
        happen when self.t is up-to-date.

        Returns
        -------
        thetaHat : float
            The regularized least squares  estimator of theta

        """
        
        _arms = self.arms[arms]
        Ainv = np.linalg.pinv(np.sum(_arms[:,None,:] * _arms[:,:,None],axis=0))
        thetaHat = Ainv @ np.sum(_arms * rewards[:,None],axis=0)
        
        return thetaHat
    
    def play(self):
        """
        Play the bandit using the optimal static allocation

        Returns
        -------
        None.

        """
        
        # Get optimal design and objective value
        optAlloc,phi = self.getOptimalAllocation()
        
        # Get number of plays per arm
        N = np.ceil(2*phi*np.log(self.Z.shape[0]/self.delta))    
        nPullsPerArm = 2*np.floor(optAlloc*N).astype(int)
        self.sampleComplexity +=  np.sum(nPullsPerArm)
        arms = np.concatenate([np.tile(i,(nPullsPerArm[i],1)) for i in range(self.n)]).squeeze()
        
        # Pull arms
        rewards = self.pull(arms)
        
        # Get (regularized) least squares estimator of theta
        thetaHat = self.estimate(arms, rewards)
        
        # Get best arm
        self.zHat = np.argmax(np.sum(self.Z * thetaHat,axis=1))
        

    def getOptimalAllocation(self,eta=1e-3,epochs=5000):
        '''
        Use the distribution of \theta_* to compute the optimal sampling allocation using mirror descent.
        '''
        X,Z,theta,initAllocation = self.arms,self.Z,self.theta,self.initAlloc
        
        arms = t.tensor(X)
        Z = t.tensor(Z)
        zStar_i = t.argmax(t.sum(Z*theta,dim=-1))
        _Z = arms[t.tensor([z for z in range(Z.shape[0]) if z!=zStar_i])]
        zStar = arms[t.argmax(t.sum(Z*theta,dim=-1))]
        
        allocation = t.tensor(initAllocation,requires_grad=True)
        
        print('Solving for optimal sampling allocation...')
        for epoch in range(epochs):
            
            # Compute objective function
            A_lambda_inv = t.inverse(t.sum(allocation[:,None,None] * (arms[:,None,:] * arms[:,:,None]),axis=0))
            diff = zStar - _Z
            objFn = t.max((diff @ A_lambda_inv @ diff.T)[0]/(diff @ theta)**2)
            
            # Compute gradient
            objFn.backward()
            grad = allocation.grad
            
            # Update using closed-form mirror descent update for KL divergence
            expAlloc = allocation * t.exp(-eta * grad)
            allocation = (expAlloc/t.sum(expAlloc)).clone().detach().requires_grad_(True)
            
            # if epoch%1000==0:
            #     print('Done with epoch {}!'.format(epoch))
        
        phi = objFn.detach().numpy()
            
        return allocation.detach().numpy(),phi

def runBenchmark():

    np.random.seed(123456)
    
    nReps = 20
    dVals = (5,10,15,20,25,30,35)
    
    sampleComplexity = []
    incorrectCount = np.zeros((len(dVals,)))
    
    for i,d in enumerate(dVals):
        
        print('d={}'.format(d))
        
        Z = np.eye(d)
        xPrime = (np.cos(.01)*Z[0] + np.sin(.01)*Z[1])[None,:]
        X = np.concatenate([Z,xPrime])
        n = X.shape[0]
        theta = 2*Z[0]
        
        T = 1000
        K = 20
        
        # Index of the optimal choice in Z
        initAlloc = np.ones((n,))/n
        
        results = []
        for rep in range(nReps):
            
            print('Replication {} of {}'.format(rep,nReps))
            
            bandit = StaticAllocation(X,Z,initAlloc,theta=theta)
            bandit.play() 
            results.append(bandit.sampleComplexity)
            
            # Check that the result is correct
            try:
                assert(np.all(bandit.arms[bandit.zHat]==X[0]))
            except:
                print('Claimed best arm: {}'.format(bandit.zHat))
                print('Best arm: {}'.format(X[0]))
                incorrectCount[i] += 1
            
        sampleComplexity.append(sum(results)/nReps)
        
    fig,ax = plt.subplots(1)
    ax.plot(dVals,sampleComplexity)
    ax.set_xlabel('d')
    ax.set_ylabel('Sample Complexity')
    ax.set_yscale('log')
    ax.set_ylim([10**2,10**8])
    ax.set_yticks([10**k for k in range(2,9)])
    
    probIncorrect = incorrectCount/nReps
    
    return sampleComplexity,probIncorrect

if __name__=='__main__':
    
    scOracle,probIncorrect = runBenchmark()
    