#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 09:02:10 2021

@author: nick
"""

import numpy as np
import torch as t
from scipy.optimize import linprog
import matplotlib.pyplot as plt

class TransductiveBandit:
    
    def __init__(self,arms,Z,initAlloc,nSteps,nRounds,lambd_reg,theta,delta=.05,eta=1e-5,sigma=1):
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
        self.nSteps = nSteps
        self.nRounds = nRounds
        
        self.allocation = np.zeros((self.nRounds+1,self.nSteps+1,self.arms.shape[0]))
        self.allocation[0,:] = initAlloc
        
        self.lambd_reg = lambd_reg
        
        # True theta used to generate rewards
        self.theta = theta
        self.zStar = np.argmax(theta)
        self._eta = eta
        self.sigma = sigma
        self.delta = delta
        
        # Dimension of the action space
        self.d = self.arms.shape[1]
        # Number of possible actions (arms)
        self.n = self.arms.shape[0]
        
        # Posterior params
        self.posteriorMean = np.zeros((self.nRounds+1,self.d))
        self.posteriorCovar = np.zeros((self.nRounds+1,self.d,self.d))
        self.posteriorCovar[0,:,:] = np.eye(self.d)
        
        # Current round
        self.k = 1
        # Current step
        self.t = 1
    
        # Store the number of times each z in \mathcal{Z} has been played here
        self.numTimesOpt = np.zeros((self.Z.shape[0],))
        
        # Store our estimates of theta here
        self.estimates = np.zeros((self.nRounds,self.nSteps,self.d))
        
        # Store z_t here
        self.z = np.zeros((self.nRounds,self.nSteps,2))
        
        # Store max objFn over current stage here
        self.maxObjFn = t.zeros((1,))
        self.maxGrad = t.zeros((self.n,))
        
        self.sampleComplexity = 0
        
        self.armsHistory = []
        self.rewardsHistory = []
        self.B_t = np.eye(self.d)
        
        self.optZ = None
        
    def eta(self):
        return self._eta / self.t**2
    
    def drawArms(self,allocation):
        
        sampledArms = np.random.choice(range(self.n),2**(self.k+1),p=allocation)
        return sampledArms
        
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
    
    # def estimate(self,arms,rewards):
    #     """
    #     Compute the regularized least squares estimator for theta. This should
    #     happen when self.t is up-to-date.

    #     Returns
    #     -------
    #     thetaHat : float
    #         The regularized least squares  estimator of theta

    #     """
        
    #     _arms = self.arms[arms]
    #     Ainv = np.linalg.pinv(np.sum(_arms[:,None,:] * _arms[:,:,None],axis=0))
    #     thetaHat = Ainv @ np.sum(_arms * rewards[:,None],axis=0)
        
    #     return thetaHat
    
    def getPosterior(self):
        """
        Compute the regularized least squares estimator for theta. This should
        happen when self.t is up-to-date.

        Returns
        -------
        thetaHat : float
            The regularized least squares  estimator of theta

        """
        
        Binv = np.linalg.pinv(self.B_t)
        thetaHat = Binv @ np.sum(np.concatenate(self.armsHistory) *\
                                 np.concatenate(self.rewardsHistory)[:,None],
                                 axis=0)
        
        return thetaHat,Binv
    
    def getBootstrapEstimates(self,arms,rewards):
        
        # Randomly choose the indices of observations we will use
        bootstrapIdx = np.array([np.random.choice(range(arms.shape[0]), arms.shape[0], replace = True) for _ in range(arms.shape[0])])
        
        thetaEstimates = np.zeros((arms.shape[0],self.d))
        
        # Get the corresponding arms and rewards
        for i in range(arms.shape[0]):
            
            bootstrapArms = arms[bootstrapIdx[i]]
            bootstrapRewards = rewards[bootstrapIdx[i]]
            
            thetaEstimates[i] = self.estimate(bootstrapArms, bootstrapRewards)
        
        return thetaEstimates
    
    def updateAllocation(self,allocation,z1,z2,takeStep=False):
        
        _arm = t.tensor(self.arms)
        allocation = t.tensor(allocation,requires_grad=True)
        
        try:
            A_lambda_inv = t.inverse(t.sum(allocation[:,None,None] * (_arm[:,None,:] * _arm[:,:,None]),axis=0))
        except Exception as e:
            print('oops')
            raise(e)
        
        z1 = t.tensor(self.arms[z1])
        z2 = t.tensor(self.arms[z2])
        diff = z1 - z2
        
        objFn = (diff.T @ A_lambda_inv @ diff)
        
        # Update running max for current stage
        if objFn > self.maxObjFn:
            
            # Use pytorch autograd to compute gradient of this expression w.r.t the allocation
            objFn.backward()
            self.maxGrad = allocation.grad
        
        if takeStep:
            
            # Take gradient step
            # newAllocation = allocation - self.eta * allocation.grad
            
            # Project back to simplex
            # newAllocation = newAllocation/t.sum(newAllocation)
            
            # Take mirror descent step
            grad = self.maxGrad
            expAlloc = allocation * t.exp(-self.eta() * grad)
            newAllocation = (expAlloc/t.sum(expAlloc)).clone()
            
            if t.any(t.isnan(newAllocation)) or t.any(newAllocation<0):
                print('oops')
    
            self.maxObjFn = t.zeros((1,))
            self.maxGrad = t.zeros((self.n,))
    
            return newAllocation.detach().numpy()
        
        else:
            return allocation.detach().numpy()
    
    def getOptimalAllocationFW(self,z1,z2,initAllocation,epochs=1000):
        '''
        Use the distribution of \theta_* to compute the optimal sampling allocation using Frank-Wolfe.
        '''
        arms = t.tensor(self.arms)
        
        allocation = t.tensor(initAllocation,requires_grad=True)
        
        # Define some stuff for minimizing inner product over simplex
        A_ub = -np.eye(self.n)
        b_ub = np.zeros((self.n,1))
        
        A_eq = np.ones((self.n,1)).T
        b_eq = 1
        
        for epoch in range(epochs):
            
            # Compute objective function
            A_lambda_inv = t.inverse(t.sum(allocation[:,None,None] * (arms[:,None,:] * arms[:,:,None]),axis=0))

            z1 = t.tensor(self.arms[z1])
            z2 = t.tensor(self.arms[z2])
            diff = z1 - z2
            
            objFn = (diff.T @ A_lambda_inv @ diff)
            
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
        
        return allocation.detach().numpy()
    
    def playStep(self):
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
        
        # Draw theta1, theta2 from posterior
        theta = np.random.multivariate_normal(self.posteriorMean[self.k-1], self.posteriorCovar[self.k-1],2)
        
        # Pick best z in Z based on draws of theta
        z1 = np.argmax(np.sum(self.Z * theta[0],axis=1))
        z2 = np.argmax(np.sum(self.Z * theta[1],axis=1))
        
        self.numTimesOpt[z1] += 1
        self.numTimesOpt[z2] += 1

        newAllocation = self.updateAllocation(self.allocation[self.k-1,self.t-1],z1,z2,
                                              takeStep=self.t>=self.nSteps)
        self.allocation[self.k-1,self.t] = newAllocation

        self.z[self.k-1,self.t-1,0] = z1
        self.z[self.k-1,self.t-1,1] = z2
        
        # Increment the step
        self.t += 1
             
        # Increment the round if we have taken enough steps
        if self.t > self.nSteps:
            
            # Draw arms from current allocation and pull
            arms = self.drawArms(self.allocation[self.k-1,self.t-1])
            _arms = self.arms[arms]
            rewards = self.pull(arms)
            
            # Track history
            self.armsHistory.append(_arms)
            self.rewardsHistory.append(rewards)
            
            # Update B matrix
            self.B_t += np.sum(_arms[:,None,:] * _arms[:,:,None],axis=0)
            
            # Update sample complexity
            self.sampleComplexity += arms.shape[0]
            
            # Compute posterior params
            mu,sigma = self.getPosterior()
            
            # posteriorCovar = np.linalg.pinv(np.linalg.pinv(self.posteriorCovar[self.k-1])+n*np.linalg.pinv(sigma))
            # posteriorMean = posteriorCovar @\
            #     (np.linalg.pinv(self.posteriorCovar[self.k-1])@self.posteriorMean[self.k-1] + n*np.linalg.pinv(sigma)@thetaBar)
                
            self.posteriorCovar[self.k] = sigma
            self.posteriorMean[self.k] = mu
            
            self.allocation[self.k,0] = newAllocation
            
            print('Done with round {}'.format(self.k))
            
            self.k += 1
            self.t = 1
            
            empiricalProbOpt = self.numTimesOpt/np.sum(self.numTimesOpt)
            # print(empiricalProbOpt)
            if np.max(empiricalProbOpt) >= 1-self.delta:
                self.optZ = np.argmax(empiricalProbOpt)
                
            self.numTimesOpt = np.zeros((self.Z.shape[0],))
        
    def play(self):
        """
        Play the bandit using LinUCB algorithm

        Returns
        -------
        None.

        """
        
        while self.k <= self.nRounds:
            
            self.playStep()
            
            if self.optZ is not None:
                break      

def getOptimalAllocation(X,Z,theta,initAllocation,eta=1e-3,epochs=5000):
    '''
    Use the distribution of \theta_* to compute the optimal sampling allocation using mirror descent.
    '''
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
        
        if epoch%1000==0:
            print('Done with epoch {}!'.format(epoch))
        
    return allocation.detach().numpy()

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
        K = 10
        
        # Index of the optimal choice in Z
        initAlloc = np.ones((n,))/n
        
        results = []
        for rep in range(nReps):
            
            print('Replication {} of {}'.format(rep,nReps))
            
            bandit = TransductiveBandit(X,Z,initAlloc,T,K,lambd_reg=0,theta=theta)
            bandit.play() 
            results.append(bandit.sampleComplexity)
            
            # Check that the result is correct
            try:
                assert(np.all(bandit.arms[bandit.optZ]==X[0]))
            except:
                print('Claimed best arm: {}'.format(bandit.optZ))
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
    
    scMax,probIncorrect = runBenchmark()
    