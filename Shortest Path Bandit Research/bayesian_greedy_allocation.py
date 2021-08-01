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

class TransductiveBandit:
    
    def __init__(self,arms,Z,initAlloc,lambd_reg,theta,delta=.05,eta=1e-5,sigma=1):
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
        self.nRounds = int(1e10)
        
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
        self.posteriorMean = np.zeros((self.d,))
        self.posteriorCovar = np.zeros((self.d,self.d))
        self.posteriorCovar = np.eye(self.d)
        
        # Current round
        self.k = 1
        
        # Base for number of samples to take in each round (e.g. self.v**self.k)
        self.v = 1
    
        # Store the number of times each z in \mathcal{Z} has been played here
        self.numTimesOpt = np.zeros((self.Z.shape[0],))
        
        # Store z_t here
        self.z = np.zeros((self.nRounds,2))
        
        self.sampleComplexity = 0
        
        self.armsHistory = []
        self.rewardsHistory = []
        self.B_t = np.eye(self.d)
        self.Binv = np.eye(self.d)
        self.rewTimesArms = np.zeros((self.d,))
        
        self.zHat = None
        
    def eta(self):
        return self._eta #/ self.k**2
    
    def drawArms(self,allocation):
        
        sampledArms = np.random.choice(range(self.n),int(np.ceil(self.v**(self.k+1))),p=allocation)
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
    
    def getNextArm(self,z1,z2):
        
        z1 = self.arms[z1]
        z2 = self.arms[z2]
        diff = z1 - z2
        
        Ainv = self.Binv
        X = self.arms
            
        _Ainv = Ainv[None,:,:] - (Ainv[None,:,:] @ (X[:,None,:] * X[:,:,None]) @ Ainv[None,:,:])/\
                (1+np.sum((X @ Ainv) *  X,axis=1))[:,None,None]
        
        # Approximate expectation of objective function using sample mean
        objFn = (diff @ _Ainv) @ diff.T
        arm = np.argmax(objFn)
        self.Binv = _Ainv[arm]

        return arm
    
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
        theta = np.random.multivariate_normal(self.posteriorMean, self.posteriorCovar, 2).T
        
        # Pick best z in Z based on draws of theta
        z1 = np.argmax(self.Z @ theta[:,0],axis=0)
        z2 = np.argmax(self.Z @ theta[:,1],axis=0)
        
        self.numTimesOpt[z1] += 1
        self.numTimesOpt[z2] += 1

        self.z[self.k-1,0] = z1
        self.z[self.k-1,1] = z2

        newArm = self.getNextArm(z1,z2)

        # Pull the arm
        reward = self.pull(newArm)
        
        # Track history 
        # self.armsHistory.append(_arms)
        # self.rewardsHistory.append(rewards)
        
        # Update B matrix
        self.updateB_t(newArm)
        
        # Update rewards times arms
        self.rewTimesArms += self.arms[newArm] * reward
        
        # Update sample complexity
        self.sampleComplexity += 1
        
        # Compute posterior params
        mu,sigma = self.getPosterior()
        
        # posteriorCovar = np.linalg.pinv(np.linalg.pinv(self.posteriorCovar[self.k-1])+n*np.linalg.pinv(sigma))
        # posteriorMean = posteriorCovar @\
        #     (np.linalg.pinv(self.posteriorCovar[self.k-1])@self.posteriorMean[self.k-1] + n*np.linalg.pinv(sigma)@thetaBar)
            
        self.posteriorCovar = sigma
        self.posteriorMean = mu
    
        if self.k % 1e5 == 0:
            print('Done with round {}'.format(self.k))
        
        # empiricalProbOpt = self.numTimesOpt/np.sum(self.numTimesOpt)
        # # print(empiricalProbOpt)
        # if np.max(empiricalProbOpt) >= 1-self.delta:
        #     self.optZ = np.argmax(empiricalProbOpt)
        
    def updateB_t(self,arm):
        
        _arm = self.arms[arm]
        self.B_t += np.outer(_arm,_arm)
        
    def checkTerminationCond(self):
        
        zHat = np.argmax(np.sum(self.Z * self.posteriorMean,axis=1))
        diff = self.Z[zHat][None,:] - self.Z
        diff = diff[np.sum(np.abs(diff),axis=1)>0]
        
        A_lambda_inv = self.Binv
        
        A_inv_norm = np.sum((diff @ A_lambda_inv[None,:,:]).squeeze() * diff,axis=1)
        
        rhs = np.sqrt(2*A_inv_norm*np.log(2*self.Z.shape[0]*self.k**2/self.delta))
        # rhs = np.sqrt(2*A_inv_norm*np.log(self.Z.shape[0]*2/self.delta))
        lhs = np.sum(diff * self.posteriorMean,axis=1)
        
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
        
        # Index of the optimal choice in Z
        initAlloc = np.ones((n,))/n
        
        results = []
        for rep in range(nReps):
            
            print('Replication {} of {}'.format(rep,nReps))
            
            bandit = TransductiveBandit(X,X,initAlloc,lambd_reg=0,theta=theta)
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
    
    scBayes,probIncorrect = runBenchmark()
    