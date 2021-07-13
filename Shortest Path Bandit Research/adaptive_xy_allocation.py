#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:32:36 2021

@author: nick
"""

import numpy as np
import torch as t
from scipy.optimize import linprog
import matplotlib.pyplot as plt

class AdaptiveXYBandit:
    
    def __init__(self,arms,Z,delta,alpha,theta,lambd_reg=0,eta=1e-5,sigma=1):
        """
        Implementation of Adaptive XY-Allocation algorithm
        https://arxiv.org/abs/1409.6110

        Parameters
        ----------
        arms : TYPE
            DESCRIPTION.
        Z : np array
            Choices (vectors) from which we are trying to maximize z^T \theta
        lambd_reg : TYPE
            DESCRIPTION.
        theta : TYPE
            DESCRIPTION.
        eta : float
            Step size for optimizing allocation
        sigma : float, optional
            Variance of Gaussian noise added to rewards. The default is 1.

        Returns
        -------
        None.

        """
        
        self.arms = arms

        # Dimension of the action space
        self.d = self.arms.shape[1]
        # Number of possible actions (arms)
        
        self.n = self.arms.shape[0]
        self.X_j = [arms,]
        self.Y = self.getY(arms)
        self.Y_j = [self.getY(arms),]
        self.rho_j = [1,]
        self.n_j = [self.d*(self.d+1)+1,]
        
        self.delta = delta
        
        self.lambd_reg = lambd_reg
        
        # True theta used to generate rewards
        self.theta = theta
        self.zStar = np.argmax(theta)
        self.eta = eta
        self.sigma = sigma
        
        self.alpha = alpha
        
        # Current round/step
        self.j = 1
        self.t = 1
        
        self.history = []
        
        self.sampleComplexity = 0
        
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
    
    def getY(self,Z):
        Y = (Z[:,None,:]-Z[None,:,:]).reshape((Z.shape[0]**2,self.d))
        Y = Y[np.abs(Y).sum(axis=1) != 0]
        
        # Only want to include differences between arms once each
        Y = np.unique(Y,axis=0)
        return Y
    
    def playStep(self,r_eps=None):
        """
        Update the state of the bandit after a step.

        Parameters
        ----------
        arm : int
            Index of the arm that was played.
        outcome : float
            The random reward which was observed.


        Returns
        -------
        None.

        """
        X_j = self.X_j[self.j-1]
        Y_j = self.Y_j[self.j-1]
        
        # Build our allocation
        A = np.eye(self.d)
        Ainv = np.copy(A)
        
        arms = []
        
        self.rho_j.append(self.rho_j[self.j-1])

        while self.rho_j[self.j]/self.t >= self.alpha *\
            self.rho_j[self.j-1] / self.n_j[self.j-1]:
            
            # Decide which arm to pull
            # _Ainv = np.linalg.pinv(A[None,:,:] + X_j[:,None,:] * X_j[:,:,None])
            _Ainv = Ainv[None,:,:] - (Ainv[None,:,:] @ (X_j[:,None,:] * X_j[:,:,None]) @ Ainv[None,:,:])/\
                (1+X_j[:,None,:] * Ainv *  X_j[:,:,None])
            
            # Should this be Y_j or Y here?...
            objFn =  np.sum(np.sum(Y_j[:,None,None,:] * _Ainv[None,:,:,:],axis=2) *\
                            Y_j[:,None,:],axis=2)
            optArm = np.argmin(np.max(objFn,axis=0))
            arms.append(optArm)
            
            A += np.outer(X_j[optArm],X_j[optArm])
            Ainv = _Ainv[optArm]
            self.t += 1
            self.rho_j[self.j] = np.max(np.sum((Y_j @ Ainv) * Y_j,axis=1))
    
        arms = np.array(arms).astype(int)
        self.history.append(arms)
        self.n_j.append(self.t-1)
        
        # Pull arms
        rewards = self.pull(arms)
        
        # Get (regularized) least squares estimator of theta
        thetaHat = self.estimate(arms, rewards)
        
        # Construct next X_j
        armsToRemove = set()
        lhsFactor = np.sqrt(np.log(self.d**2/self.delta)/np.log(self.n_j[self.j]))
        for arm in X_j:
            diff = arm[None,:] - X_j
            diff = diff[np.sum(np.abs(diff),axis=1)>0]
            lhs = np.sum((diff @ Ainv) * diff,axis=1) * lhsFactor
            DeltaHat_j = np.sum(-diff * thetaHat,axis=1)
            if np.any(lhs <= DeltaHat_j):
                armsToRemove.add( tuple(arm) )
        
        _X_j = set([tuple(arm) for arm in X_j]) - armsToRemove
        _X_j = np.array([np.array(item) for item in _X_j])
        self.X_j.append(_X_j)
        
        if _X_j.shape[0] > 1:    
            self.Y_j.append(self.getY(self.X_j[-1]))
        
        # Increment the round and reset step to 1
        # print('Done with round {}'.format(self.j))
        # print('Number of arms left: {}'.format(_X_j.shape[0]))
        self.j += 1
        self.t = 1
        
    def play(self):
        """
        Play the bandit using LinUCB algorithm

        Returns
        -------
        None.

        """
        
        while self.X_j[self.j-1].shape[0] > 1:
            
            self.playStep()
            self.sampleComplexity += self.t-1  
        
def runBenchmark():

    np.random.seed(123456)
    
    nReps = 20
    dVals = (5,10,15,20,25,30,35)
    # dVals = (20,)
    
    sampleComplexity = []
    incorrectCount = np.zeros((len(dVals,)))
    
    for i,d in enumerate(dVals):
        
        print('d={}'.format(d))
        
        Z = np.eye(d)
        xPrime = (np.cos(.01)*Z[0] + np.sin(.01)*Z[1])[None,:]
        X = np.concatenate([Z,xPrime])
        n = X.shape[0]
        theta = 2*Z[0]
        
        # Index of the optimal choice in Z
        initAlloc = np.ones((n,))/n
        
        results = []
        for rep in range(nReps):
            
            print('Replication {} of {}'.format(rep,nReps))
            
            bandit = AdaptiveXYBandit(X,Z,delta=.05,alpha=.1,theta=theta)
            bandit.play() 
            results.append(bandit.sampleComplexity)
            
            # Check that the result is correct
            try:
                assert(np.all(bandit.X_j[-1]==X[0]))
            except:
                print('Claimed best arm: {}'.format(bandit.X_j[-1]))
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
    
    return bandit,probIncorrect
 
if __name__=='__main__':
    
    # bandit,probIncorrect = runTransductiveExample()
    sc,probIncorrect = runBenchmark()
    