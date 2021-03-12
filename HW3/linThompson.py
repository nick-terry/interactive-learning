#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:32:18 2021

@author: nick
"""

import numpy as np
import mnistLoader as mn

class LinThompsonBandit:
    
    def __init__(self,arms,horizon,lambd_reg,contextPool,labelPool,sigma=1):
        """
        Implementation of the linear UCB algorithm for Multi-armed bandits

        Parameters
        ----------
        arms : TYPE
            DESCRIPTION.
        horizon : TYPE
            DESCRIPTION.
        sigma : float, optional
            Variance of Gaussian noise added to rewards. The default is 1.
            
        Returns
        -------
        None.

        """
        
        self.arms = arms
        self.horizon  = horizon
        # True theta used to generate rewards
        self.sigma = sigma
        
        # Number of possible actions (arms)
        self.n = self.arms.shape[0]
        # Pool from which contexts can be drawn.
        self.contextPool = contextPool
        self.labelPool = labelPool
        
        # Dimension of the context space
        self.d = self.contextPool.shape[1] * self.n
        
        # V matrix used to compute estimate of theta
        self.V = lambd_reg * np.eye(self.d)
        self.VInv = np.linalg.inv(self.V)
        
        # Store the contexts and labels observed here
        # self.C = np.zeros((self.horizon,self.contextPool.shape[1]))
        cI = np.random.choice(range(self.contextPool.shape[0]),size=self.horizon)
        self.C = self.contextPool[cI]
        self.Labels = self.labelPool[cI]
        
        # Keep running sum of rewards times feature vectors
        self.rewardsTimesFeats = np.zeros((self.d,))
        
        # Current round
        self.t = 1
        
        # Define prior parameters
        self.priorMean = np.zeros((self.d,))
        self.priorCovar = np.eye(self.d)
        
        # Store posterior params here. We are using MVG prior w/ assumed known identity covar
        self.posteriorMeans = np.zeros((self.horizon,self.d))
        self.posteriorCovars = np.zeros((self.horizon,self.d,self.d))
        self.posteriorMeans[0,:] = self.priorMean
        self.posteriorCovars[0,:] = self.priorCovar
    
        # Store the number of times each arms has been played here
        self.num_plays = np.zeros((self.n,))
        
        # Store our estimates of theta here
        self.estimates = np.zeros((self.horizon,self.d))
        
        # Store the rewards here
        self.rewards = np.zeros((self.horizon,))
        
        # Record which arm is pulled in each round here
        self.history = np.zeros(shape=(self.horizon,)).astype(int)
        
        # Record regret at each round here
        self.regret = np.zeros(shape=(self.horizon,))
        
        # Compute the maximum possible reward (for computing regret)
        self.opt_reward = 1
        
    def pull(self,context,arm,label):
        """
        Pull arm and generate random reward

        Parameters
        ----------
        arm : int
            Index of the arm to pull

        Returns
        -------
        outcome : float
            The random reward.

        """
        
        action = self.arms[arm]
        outcome = action==label
        
        return outcome
    
    def estimateTheta(self):
        """
        Compute the regularized least squares estimator for theta. This should
        happen when self.t is up-to-date.

        Returns
        -------
        thetaHat : float
            The regularized least squares  estimator of theta

        """
        
        # From equation 19.5 in Szepsvari, Lattimore
        thetaHat = np.linalg.solve(self.V,self.rewardsTimesFeats)
        
        return thetaHat
    
    def chooseArm(self,c):
        """
        Choose the best arm by drawing theta_t from posterior and choosing action
        which maximizes reward.

        Returns
        -------
        arm : int
            Index of the best arm to play in current round

        """
        
        posteriorMean = self.estimates[self.t-2]
        # posteriorCovar = np.linalg.inv(self.V)
        posteriorCovar = self.VInv
        
        self.posteriorMeans[self.t-1,:] = posteriorMean
        self.posteriorCovars[self.t-1,:,:] = posteriorCovar
        
        # Draw theta_t from posterior
        theta_t = np.random.multivariate_normal(posteriorMean, posteriorCovar, size=1)
        
        repC = np.repeat(c[None,:],self.n,axis=0)
        feat = np.reshape(repC[:,None,:] * np.eye(self.n)[:,:,None],
                          (self.n,self.d))
        
        # Choose arm which maximizes inner product with theta_t
        objFn = np.sum(feat * theta_t, axis=1)
        optArm = np.argmax(objFn)
        
        return optArm
    
    def update(self,context,arm,reward):
        """
        Update the state of the bandit after a round.

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
        feat = self.getFeature(context, arm)
        self.rewardsTimesFeats += feat * reward
        
        # Update V matrix and its inverse
        B = np.outer(feat,feat)
        self.V += B
        # Invert the new VInv using a neat trick: https://math.stackexchange.com/questions/17776/inverse-of-the-sum-of-matrices
        self.VInv = self.VInv - (self.VInv @ B @ self.VInv)/(1+np.trace(B @ self.VInv))
        
        # Compute new estimate of thetae
        thetaHat = self.estimateTheta()
        self.estimates[self.t-1] = thetaHat
        
        # Update the state of the bandit
        self.history[self.t-1] = arm
        self.num_plays[arm] += 1
        self.rewards[self.t-1] = reward
        self.regret[self.t-1] = self.opt_reward - reward
        
        
        # Increment the round
        self.t += 1
        if self.t%1000 == 0:
            print(self.t)
    
    def getFeature(self,context,arm):
        
        return (context[:,None] * np.eye(self.n)[arm,:][None,:]).T.reshape(self.d)
    
    def play(self):
        """
        Play the bandit using LinUCB algorithm

        Returns
        -------
        None.

        """
        
        while self.t <= self.horizon:
            
            context,label = self.C[self.t-1],self.Labels[self.t-1]
            
            arm = self.chooseArm(context)
            reward = self.pull(context,arm,label)
            self.update(context,arm,reward)
        
        
if __name__=='__main__':
    
    np.random.seed(1234)
    
    d = 100
    n = 100
    T = 50000
    
    X_train,y_train,X_test,y_test = mn.loadMNIST()
    contextPool = mn.rescale(mn.getRepresentation(X_train,d=16))
    labelPool = y_train
    X = np.unique(y_train)
    
    bandit = LinThompsonBandit(arms=X, horizon=T, lambd_reg=1,
                          contextPool=contextPool, labelPool=labelPool)
    bandit.play()