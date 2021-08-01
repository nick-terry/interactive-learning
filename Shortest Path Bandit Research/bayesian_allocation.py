#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:05:21 2021

@author: nick
"""
import sys
import numpy as np
import torch as t
from scipy.optimize import linprog
import matplotlib.pyplot as plt

t.autograd.set_detect_anomaly(False)

class TransductiveBandit:
    
    def __init__(self,arms,Z,initAlloc,nSteps,nRounds,lambd_reg,theta,delta=.05,eta=1e-3,sigma=1,v=2):
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
        
        self.allocation = np.zeros((self.nRounds+1,self.arms.shape[0]))
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
        
        # Base for number of samples to take in each round (e.g. self.v**self.k)
        self.v = v
    
        # Store the number of times each z in \mathcal{Z} has been played here
        self.numTimesOpt = np.zeros((self.Z.shape[0],))
        
        # Store our estimates of theta here
        self.estimates = np.zeros((self.nRounds,self.nSteps,self.d))
        self.armPlays = np.zeros((self.n,))
        
        # Store z_t here
        self.z = -np.ones((self.nRounds,self.nSteps,2))
        
        self.sampleComplexity = 0
        
        self.armsHistory = []
        self.rewardsHistory = []
        self.B_t = np.eye(self.d)
        self.rewTimesArms = np.zeros((self.d,))
        
        self.zHat = None
        
    def eta(self):
        return self._eta #/ self.k**2
    
    def drawArms(self,allocation):
        
        sampledArms = np.random.choice(range(self.n),int(np.ceil(self.v**(self.k+2))),p=allocation)
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
        # thetaHat = Binv @ np.sum(np.concatenate(self.armsHistory) *\
        #                          np.concatenate(self.rewardsHistory)[:,None],
        #                          axis=0)
        thetaHat = Binv @ self.rewTimesArms
        
        return thetaHat,Binv
    
    def updateAllocation(self,allocation,z1,z2,nSteps=5000):
        
        _arm = t.tensor(self.arms)
        
        z1 = t.tensor(self.arms[z1])
        z2 = t.tensor(self.arms[z2])
        diff = z1 - z2
        
        objFnArray = t.zeros((nSteps,))
        
        allocation = t.tensor(allocation,requires_grad=True)
        
        for step in range(1,nSteps+1):
            
            try:
                A_lambda_inv = t.inverse(t.tensor(self.B_t) + t.sum(int(np.ceil(self.v**(self.k+1))) *\
                                               allocation[:,None,None] * (_arm[:,None,:] * _arm[:,:,None]),axis=0))
            except Exception as e:
                print('oops')
                raise(e)
            
            # Approximate expectation of objective function using sample mean
            objFn = t.max(t.sum((diff @ A_lambda_inv) * diff, dim=1))
            objFnArray[step-1] = objFn
            
            # Use pytorch autograd to compute gradient of this expression w.r.t the allocation
            objFn.backward()
            
            # Take mirror descent step
            grad = allocation.grad
            expAlloc = allocation * t.exp(-self.eta() * grad)
            newAllocation = (expAlloc/t.sum(expAlloc)).clone()

            Delta = t.norm(newAllocation - allocation,p=2)
            allocation = newAllocation.clone().detach().requires_grad_(True)
            
            # if Delta < .01:
            #     break
        # print(allocation)
        return allocation.detach().numpy()
    
    def updateAllocationFW(self,allocation,z1,z2,epochs=1000):
        
        z1 = t.tensor(self.Z[z1])
        z2 = t.tensor(self.Z[z2])
        diff = z1 - z2
        diff = t.unique(diff,dim=0)
        
        allocation = t.tensor(allocation,requires_grad=True)
        
        X = t.tensor(self.arms)
        
         # Define some stuff for minimizing inner product over simplex
        A_ub = -np.eye(self.n)
        b_ub = np.zeros((self.n,1))
        
        A_eq = np.ones((self.n,1)).T
        b_eq = 1
        
        yHatList = []
        for epoch in range(epochs):

            # Compute objective function
            # A_lambda_inv = t.inverse(t.sum(t.tensor(self.B_t) +\
            #                                    self.v**(self.k+1) *\
            #                                    allocation[:,None,None] * (X[:,None,:] * X[:,:,None]),axis=0))
            try:
                A_lambda_inv = t.inverse(t.sum(allocation[:,None,None] * (X[:,None,:] * X[:,:,None]),axis=0))
            except Exception as e:
                A_lambda_inv = t.pinverse(t.sum(allocation[:,None,None] * (X[:,None,:] * X[:,:,None]),axis=0))
            # diff = (X[None,:,:] - X[:,None,:]).reshape(X.shape[0]**2,X.shape[1])
            # diff = t.unique(diff,dim=0)
            # objFn = t.max(t.sum((diff @ A_lambda_inv) * diff, dim=1))
            objFn = t.max(t.sum(diff @ A_lambda_inv  * diff, dim=1))
            yHat = t.argmax(t.sum(diff @ A_lambda_inv  * diff, dim=1))
            yHatList.append(yHat)
            
            # Compute gradient
            # objFn.backward()
            # grad = allocation.grad
            
            grad = (-diff[yHat] @ A_lambda_inv @ (X[:,None,:] * X[:,:,None]) @ A_lambda_inv @ diff[yHat]).detach()
            
            # if t.any(grad.isnan()):
            #     break
            
            # Update using Frank-Wolfe step
            try:
                aMin = t.tensor(linprog(grad.numpy(),A_ub,b_ub,A_eq,b_eq).x)
            except Exception as e:
                print(e)
            gamma = 2/(epoch+2)
            _allocation = (1-gamma) * allocation + gamma * aMin

            Delta = t.norm(_allocation - allocation,p=2)
            allocation = _allocation.clone().detach().requires_grad_(True)
            
            # if Delta < .01:
            #     break

        with t.no_grad():

            toZeroMask = allocation<10**-5
            allocation[toZeroMask] = 0
            allocation = allocation/t.sum(allocation)
        
        # print('Finished in {} epochs'.format(epoch))
        
        # print(allocation.detach().numpy())
        # print(np.bincount(self.z[self.k-1,:].reshape((self.nSteps*2)).astype(int),minlength=self.Z.shape[0]))
        # print('y\'s used to optimize:')
        # yUnique = np.unique(np.stack(yHatList),axis=0)
        # print(diff.numpy()[yUnique].astype(np.int))
        # print('y counts:')
        # yCounts = np.bincount(np.stack(yHatList),minlength=yUnique.shape[0])
        # diffnp = diff.numpy().astype(int)
        # for y in yUnique:
        #     print('y: {} was max {} times'.format(diffnp[y],yCounts[y]))
        
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
        
        # Draw T pairs theta1, theta2 from posterior
        theta = np.random.multivariate_normal(self.posteriorMean[self.k-1], self.posteriorCovar[self.k-1], 2*self.nSteps).T
        
        # Pick best z in Z based on draws of theta
        z1 = np.argmax(self.Z @ theta[:,:self.nSteps],axis=0)
        z2 = np.argmax(self.Z @ theta[:,self.nSteps:],axis=0)
        
        self.numTimesOpt += np.bincount(z1, minlength=self.Z.shape[0])
        self.numTimesOpt += np.bincount(z2, minlength=self.Z.shape[0])

        self.z[self.k-1,:,0] = z1
        self.z[self.k-1,:,1] = z2

        newAllocation = self.updateAllocationFW(self.allocation[self.k-1],z1,z2)
        self.allocation[self.k-1] = newAllocation
            
        # Draw arms from current allocation and pull
        arms = self.drawArms(self.allocation[self.k-1])
        rewards = self.pull(arms)
        
        # Track history 
        # self.armsHistory.append(_arms)
        # self.rewardsHistory.append(rewards)
        
        # Update B matrix
        self.updateB_t(arms)
        
        # Update rewards times arms
        self.rewTimesArms += np.sum(self.arms[arms] * rewards[:,None],axis=0)
        
        # Update sample complexity
        self.sampleComplexity += arms.shape[0]
        
        # Compute posterior params
        mu,sigma = self.getPosterior()
        
        # posteriorCovar = np.linalg.pinv(np.linalg.pinv(self.posteriorCovar[self.k-1])+n*np.linalg.pinv(sigma))
        # posteriorMean = posteriorCovar @\
        #     (np.linalg.pinv(self.posteriorCovar[self.k-1])@self.posteriorMean[self.k-1] + n*np.linalg.pinv(sigma)@thetaBar)
            
        self.posteriorCovar[self.k] = sigma
        self.posteriorMean[self.k] = mu
        
        self.allocation[self.k] = newAllocation
        
        print('Done with round {}'.format(self.k))
        
        # empiricalProbOpt = self.numTimesOpt/np.sum(self.numTimesOpt)
        # # print(empiricalProbOpt)
        # if np.max(empiricalProbOpt) >= 1-self.delta:
        #     self.optZ = np.argmax(empiricalProbOpt)
        
    def updateB_t(self,arms):
        
        playsPerArm = np.bincount(arms,minlength=self.n)
        
        self.armPlays += playsPerArm
        self.B_t += np.sum(playsPerArm[:,None,None] * (self.arms[:,None,:] * self.arms[:,:,None]),axis=0)
        
    def checkTerminationCond(self):
        
        zHat = np.argmax(np.sum(self.Z * self.posteriorMean[self.k],axis=1))
        diff = self.Z[zHat][None,:] - self.Z
        diff = diff[np.sum(np.abs(diff),axis=1)>0]
        
        # allocation = self.allocation[self.k]
        # A_lambda_inv = np.linalg.pinv(self.sampleComplexity * np.sum(allocation[:,None,None] * (self.arms[:,None,:] * self.arms[:,:,None]),axis=0))
        A_lambda_inv = np.linalg.pinv(self.B_t)
        
        A_inv_norm = np.sum((diff @ A_lambda_inv[None,:,:]).squeeze() * diff,axis=1)
        
        rhs = np.sqrt(2*A_inv_norm*np.log(2*self.Z.shape[0]*self.k**2/self.delta))
        # rhs = np.sqrt(2*A_inv_norm*np.log(self.Z.shape[0]*2/self.delta))
        lhs = np.sum(diff * self.posteriorMean[self.k],axis=1)
        
        # print(lhs - rhs)
        terminateNow = np.all(lhs >= rhs)
        
        return terminateNow,zHat
    
    def play(self):
        """
        Play the bandit using LinUCB algorithm

        Returns
        -------
        None.

        """
        
        terminateNow = False
        
        while not terminateNow:
            
            zHat = self.playStep()
            
            # Check if termination condition is met
            terminateNow,zHat = self.checkTerminationCond()  
            self.zHat = zHat
            
            self.k += 1

def runBenchmark():

    np.random.seed(123456)
    
    nReps = 20
    # dVals = (5,10,15,20,25,30)
    # dVals = (5,10,15,20,25,30,35)
    dVals = (5,)
    
    sampleComplexity = []
    incorrectCount = np.zeros((len(dVals,)))
    
    for i,d in enumerate(dVals):
        
        print('d={}'.format(d))
        
        Z = np.eye(d)
        xPrime = (np.cos(.01)*Z[0] + np.sin(.01)*Z[1])[None,:]
        X = np.concatenate([Z,xPrime])
        n = X.shape[0]
        theta = 2*Z[0]
        
        T = 2500
        K = 1000
        
        # Index of the optimal choice in Z
        initAlloc = np.ones((n,))/n
        
        results = []
        for rep in range(nReps):
            
            print('Replication {} of {}'.format(rep,nReps))
            
            bandit = TransductiveBandit(X,X,initAlloc,T,K,lambd_reg=0,theta=theta,v=2)
            bandit.play() 
            results.append(bandit.sampleComplexity)
            
            # Check that the result is correct
            try:
                assert(np.all(bandit.arms[bandit.zHat]==X[0]))
            except:
                print('Claimed best arm: {}'.format(bandit.zHat))
                print('Best arm: {}'.format(X[0]))
                incorrectCount[i] += 1
            
        sampleComplexity.append(sum(results)/len(results))
        
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
    
    scBayes,probIncorrect = runBenchmark()
    