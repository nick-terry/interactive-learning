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

class OracleAllocation:
    
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
        
        self.A = np.eye(self.d)
        self.Ainv = np.eye(self.d)
        self.b = np.zeros((self.d,))
        
        self.sampleComplexity = 0
        
        self.zHat = None
        
        self.t = 1
        
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
    
    def getY(self,X):
        Y = (X[:,None,:]-X[None,:,:]).reshape((X.shape[0]**2,self.d))
        Y = Y[np.abs(Y).sum(axis=1) != 0]
        
        # Only want to include differences between arms once each
        Y = np.unique(Y,axis=0)
        return Y
    
    def drawArms(self,allocation):
        
        sampledArms = np.random.choice(np.arange(self.n),2**(self.t+1),p=allocation)
        return sampledArms
    
    def playStep(self):
        
        X = self.arms
        
        # # Choose next arm to greedily add
        # _Ainv = Ainv[None,:,:] - (Ainv[None,:,:] @ (X[:,None,:] * X[:,:,None]) @ Ainv[None,:,:])/\
        #         (1+X[:,None,:] * Ainv *  X[:,:,None])
            
        # objFn =  np.sum(np.sum(Y[:,None,None,:] * _Ainv[None,:,:,:],axis=2) *\
        #                 Y[:,None,:],axis=2)
        # optArmToAdd = np.argmin(np.max(objFn,axis=0))
        
        # Draw arms to pull from sampling allocation
        arms = self.drawArms(self.alloc)
        
        # Pull arms
        rewards = self.pull(arms)
        
        # Update some stored values
        self.updateA(arms)
        self.Ainv = np.linalg.pinv(self.A)
        self.b += np.sum(self.arms[arms] * rewards[:,None],axis=0)
        
        self.sampleComplexity += arms.shape[0]
        
        # Get (regularized) least squares estimator of theta
        self.thetaHat = self.Ainv @ self.b
        
        # Get our guess of the best arm
        self.zHat = np.argmax(np.sum(self.Z * self.thetaHat,axis=1)).astype(int)
        
        self.t += 1
        if self.t %10000 ==0:
            print('Done with round {}'.format(self.t))
    
    def updateA(self,arms):
        
        playsPerArm = np.bincount(arms,minlength=self.n)
        
        self.A += np.sum(playsPerArm[:,None,None] * (self.arms[:,None,:] * self.arms[:,:,None]),axis=0)
    
    def checkTerminatingCond(self):
         
        zHat = self.arms[self.zHat]
        diff = zHat[None,:] - self.Z
        diff = diff[np.sum(np.abs(diff),axis=1)>0]
        
        A_lambda_inv = np.linalg.inv(self.sampleComplexity * np.sum(self.alloc[:,None,None] * (self.arms[:,None,:] * self.arms[:,:,None]),axis=0))
        A_inv_norm = np.sum((diff @ A_lambda_inv[None,:,:]).squeeze() * diff,axis=1)
       
        # rhs = 2*np.sqrt(2*A_inv_norm*np.log(self.arms.shape[0]**2/self.delta)/np.log(self.t))
        
        # Use rhs from RAGE
        rhs = np.sqrt(2*A_inv_norm*np.log(2*self.arms.shape[0] * self.t**2/self.delta))
        lhs = np.sum(diff * self.thetaHat,axis=1)
        # print(lhs-rhs)
        terminateNow = np.all(lhs >= rhs)
        
        return terminateNow
    
    def play(self):
        """
        Play the bandit using the optimal static allocation

        Returns
        -------
        None.

        """
        terminate = False
        
        self.alloc,_ = self.getOracleAllocationFW()
        
        while not terminate:
            
            self.playStep()
            terminate = self.checkTerminatingCond()
    
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
    
    def getOracleAllocationFW(self,eta=1e-3,epochs=5000):
        '''
        Use the distribution of \theta_* to compute the optimal sampling allocation using Frank-Wolfe.
        '''
        X,Z,theta,initAllocation = self.arms,self.Z,self.theta,self.initAlloc
        
        arms = t.tensor(X)
        Z = t.tensor(Z)
        zStar_i = t.argmax(t.sum(Z*theta,dim=-1))
        _Z = arms[t.tensor([z for z in range(Z.shape[0]) if z!=zStar_i])]
        zStar = arms[t.argmax(t.sum(Z*theta,dim=-1))]
        
        allocation = t.tensor(initAllocation,requires_grad=True)
        
        # Define some stuff for minimizing inner product over simplex
        A_ub = -np.eye(self.n)
        b_ub = np.zeros((self.n,1))
        
        A_eq = np.ones((self.n,1)).T
        b_eq = 1
        
        for epoch in range(epochs):

            # Compute objective function
            A_lambda_inv = t.inverse(t.sum(allocation[:,None,None] * (arms[:,None,:] * arms[:,:,None]),axis=0))
            diff = zStar - _Z
            objFn = t.max((diff @ A_lambda_inv @ diff.T)[0]/(diff @ theta)**2)
            
            # Compute gradient
            objFn.backward()
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
        
        return allocation.detach().numpy(),objFn.detach().numpy()

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
            
            bandit = OracleAllocation(X,X,initAlloc,theta=theta)
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
    