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

class RageBandit:
    
    def __init__(self,arms,Z,delta,epsilon,theta,initAlloc,lambd_reg=0,eta=1e-5,sigma=1):
        """
        Implementation of Rage Algorithm
        https://arxiv.org/abs/1906.08399

        Parameters
        ----------
        arms : TYPE
            DESCRIPTION.
        Z : np array
            Choices (vectors) from which we are trying to maximize z^T \theta
        initAlloc : np array
            Initial allocation over the arms
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
        self.Z = Z
        self.Z_t = [Z,]
        self.delta = delta
        self.epsilon = epsilon
        
        self.initAlloc = initAlloc
        
        self.lambd_reg = lambd_reg
        
        # True theta used to generate rewards
        self.theta = theta
        self.zStar = np.argmax(theta)
        self.eta = eta
        self.sigma = sigma
        
        # Dimension of the action space
        self.d = self.arms.shape[1]
        # Number of possible actions (arms)
        self.n = self.arms.shape[0]
        
        # Current round
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
    
    def fastRound(self,alloc,N):
        
        l = 1
        
        p = alloc.shape[0]
        N_alloc = np.ceil((N-.5*p)*alloc)
        diff = np.sum(N_alloc) - N
        
        while np.sum(N_alloc) - N != 0:
            # minEntry = np.min(N_alloc/alloc)
            # maxEntry = np.max((N_alloc-1)/alloc)
            
            # minMask = N_alloc/alloc==minEntry
            # maxMask = (N_alloc-1)/alloc==maxEntry
            
            # if np.sum(minMask)>0:
            #     j = np.where(minMask)[0][0]
            #     N_alloc[j] = N_alloc[j] + 1
                
            # elif np.sum(maxMask)>0:
            #     k = np.where(maxMask)[0][0]
            #     N_alloc[k] = N_alloc[k] - 1
            
            # We might divide by zero here
            with np.errstate(divide='ignore'):
                j = np.argmin(np.nan_to_num(N_alloc/alloc,nan=np.inf))
                k = np.argmax(np.nan_to_num((N_alloc-1)/alloc,nan=0))

            if diff<0:
                N_alloc[j] = N_alloc[j] + 1
            else:
                N_alloc[k] = N_alloc[k] - 1
        
        return N_alloc.astype(int)
    
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
        if r_eps is None:
            # r_eps=(self.d*(self.d+1)/2 +1)/self.epsilon
            r_eps=self.d**2/self.epsilon
        
        delta_t = self.delta/self.t**2
        Z_t = self.Z_t[self.t-1]
        
        # Get the optimal sampling allocation
        lambda_t,rho_t = self.getOptimalAllocationFW(Z_t, self.initAlloc)
        
        # Compute the total number of samples to take for this round
        N_t = np.maximum(np.ceil(8*(2**(self.t+1))**2 * rho_t *\
                                 (1+self.epsilon) * np.log(self.Z.shape[0]**2 / delta_t)),
                                 r_eps)
            
        self.sampleComplexity += N_t
        
        # Round the allocation to get the number of times to pull each arm
        nPullsPerArm = self.fastRound(lambda_t,N_t)
        arms = np.concatenate([np.tile(i,(nPullsPerArm[i],1)) for i in range(self.n)]).squeeze()
    
        # Pull arms
        rewards = self.pull(arms)
        
        # Get (regularized) least squares estimator of theta
        thetaHat = self.estimate(arms, rewards)
        
        # Figure out which arms are eliminated
        armsToRemove = set()
        for arm in Z_t:
            innerProd = np.sum((Z_t - arm[None,:]) * thetaHat,axis=1)
            if np.any(innerProd >= 2** -(self.t+2)):
                armsToRemove.add( tuple(arm) )
        
        _Z_t = set([tuple(arm) for arm in Z_t]) - armsToRemove
        self.Z_t.append(np.array([np.array(item) for item in _Z_t]))
        
        # Increment the step
        self.t += 1
        print('Done with round {}'.format(self.t-1))
        
    def play(self):
        """
        Play the bandit using LinUCB algorithm

        Returns
        -------
        None.

        """
        
        while self.Z_t[self.t-1].shape[0] > 1:
            
            self.playStep()
        

    def getOptimalAllocation(self,Z_t,initAllocation,epochs=2500,eta=1e-3):
        '''
        Use the distribution of \theta_* to compute the optimal sampling allocation using mirror descent.
        '''
        arms = t.tensor(self.arms)
        Z = t.tensor(Z_t)
        
        Y = t.tensor(self.getY(Z_t))
        
        allocation = t.tensor(initAllocation,requires_grad=True)
        
        for epoch in range(epochs):
            
            # Compute objective function
            A_lambda_inv = t.inverse(t.sum(allocation[:,None,None] * (arms[:,None,:] * arms[:,:,None]),axis=0))

            rho = t.max((Y @ A_lambda_inv @ Y.T)[0])
            
            # Compute gradient
            rho.backward()
            grad = allocation.grad
            
            # Update using closed-form mirror descent update for KL divergence
            expAlloc = allocation * t.exp(-eta * grad)
            allocation = (expAlloc/t.sum(expAlloc)).clone().detach().requires_grad_(True)

            
        return allocation.detach().numpy(),rho.detach().numpy()
    
    def getOptimalAllocationFW(self,Z_t,initAllocation,epochs=1000):
        '''
        Use the distribution of \theta_* to compute the optimal sampling allocation using Frank-Wolfe.
        '''
        arms = t.tensor(self.arms)
        Z = t.tensor(Z_t)
        
        Y = t.tensor(self.getY(Z_t))
        
        allocation = t.tensor(initAllocation,requires_grad=True)
        
        # Define some stuff for minimizing inner product over simplex
        A_ub = -np.eye(self.n)
        b_ub = np.zeros((self.n,1))
        
        A_eq = np.ones((self.n,1)).T
        b_eq = 1
        
        for epoch in range(epochs):
            
            # Compute objective function
            A_lambda_inv = t.inverse(t.sum(allocation[:,None,None] * (arms[:,None,:] * arms[:,:,None]),axis=0))

            rho = t.max((Y @ A_lambda_inv @ Y.T)[0])
            
            # Compute gradient
            rho.backward()
            grad = allocation.grad
            
            # Update using Frank-Wolfe step
            aMin = t.tensor(linprog(grad.numpy(),A_ub,b_ub,A_eq,b_eq).x)
            gamma = 2/(epoch+2)
            _allocation = (1-gamma) * allocation + gamma * aMin

            Delta = t.norm(_allocation - allocation,p=2)
            allocation = _allocation.clone().detach().requires_grad_(True)
            
            if Delta < .01:
                break

        with t.no_grad():

            toZeroMask = allocation<10**-5
            allocation[toZeroMask] = 0
            allocation = allocation/t.sum(allocation)
        
        return allocation.detach().numpy(),rho.detach().numpy()
        
def runBenchmark():

    np.random.seed(123456)
    
    nReps = 20
    dVals = (5,10,15,20,25,30,35)
    # dVals = (35,)
    
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
            
            bandit = RageBandit(X,Z,delta=.05,epsilon=.2,theta=theta,initAlloc=initAlloc)
            bandit.play() 
            results.append(bandit.sampleComplexity)
            
            # Check that the result is correct
            try:
                assert(np.all(bandit.Z_t[-1]==X[0]))
            except:
                print('Claimed best arm: {}'.format(bandit.Z_t[-1]))
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
    
def runTransductiveExample():
    
    np.random.seed(12345)
    
    nReps = 20
    dVals = (20,40,60,80)
    
    # nReps = 100
    # dVals = (4,)
    
    sampleComplexity = []
    
    incorrectCount = 0
    
    for d in dVals:
        
        print('d={}'.format(d))
        
        X = np.eye(d)
        Z = np.concatenate([np.eye(d)[:d//2],] +\
                           [(np.cos(.1)*X[j] + np.sin(.1)*X[j+d//2])[None,:] for j in range(0,d//2)])

        n = X.shape[0]
        theta = X[0]
        
        # Index of the optimal choice in Z
        initAlloc = np.ones((n,))/n
        
        results = []
        for rep in range(nReps):
            
            print('Replication {} of {}'.format(rep,nReps))
            
            bandit = RageBandit(X,Z,delta=.05,epsilon=.2,theta=theta,initAlloc=initAlloc)
            bandit.play()
            
            # Check that the result is correct
            try:
                assert(np.all(bandit.Z_t[-1]==X[0]))
            except:
                print('Claimed best arm: {}'.format(bandit.Z_t[-1]))
                print('Best arm: {}'.format(X[0]))
                incorrectCount += 1
                
            results.append(bandit.sampleComplexity)
            
        sampleComplexity.append(sum(results)/nReps)
        
    fig,ax = plt.subplots(1)
    ax.plot(dVals,sampleComplexity)
    ax.set_xlabel('d')
    ax.set_ylabel('Sample Complexity')
    ax.set_yscale('log')
    ax.set_ylim([10**2,10**8])
    ax.set_yticks([10**k for k in range(2,9)])
    
    probIncorrect = incorrectCount/(nReps*len(dVals))
    
    return sampleComplexity,probIncorrect

if __name__=='__main__':
    
    # bandit,probIncorrect = runTransductiveExample()
    sc,probIncorrect = runBenchmark()
    