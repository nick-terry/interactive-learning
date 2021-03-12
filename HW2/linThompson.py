#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:32:18 2021

@author: nick
"""

import numpy as np

class LinThompsonBandit:
    
    def __init__(self,arms,horizon,theta,sigma=1,L=1,m2=1):
        """
        Implementation of the linear UCB algorithm for Multi-armed bandits

        Parameters
        ----------
        arms : TYPE
            DESCRIPTION.
        horizon : TYPE
            DESCRIPTION.
        theta : TYPE
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
        self.theta = theta
        self.sigma = sigma
        
        # Dimension of the action space
        self.d = self.arms.shape[1]
        # Number of possible actions (arms)
        self.n = self.arms.shape[0]
        # V matrix used to compute estimate of theta
        self.V = np.eye(self.d)
        self.VInv = np.linalg.inv(self.V)
        
        # Track running sum of rewards times arms
        self.rewardsTimesArms = np.zeros((self.d,))
        
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
        self.opt_reward = np.max(np.sum(self.arms * self.theta, axis=1))
        
    def pull(self,arm):
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
        outcome = np.dot(action,self.theta) + np.random.normal(0,self.sigma**2)
        
        return outcome
    
    def estimate(self):
        """
        Compute the regularized least squares estimator for theta. This should
        happen when self.t is up-to-date.

        Returns
        -------
        thetaHat : float
            The regularized least squares  estimator of theta

        """
        
        # From equation 19.5 in Szepsvari, Lattimore
        # b = np.sum(self.rewards[:self.t-1,None]*self.arms[self.history[:self.t-1]],axis=0)
        thetaHat = self.VInv @ self.rewardsTimesArms
        
        return thetaHat
    
    def chooseArm(self):
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
        
        # Choose arm which maximizes inner product with theta_t
        objFn = np.sum(self.arms * theta_t, axis=1)
        optArm = np.argmax(objFn)
        
        return optArm
    
    def update(self,arm,outcome):
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
        _arm = self.arms[arm]
        
        # Update V matrix and its inverse
        B = np.outer(_arm,_arm)
        self.V += B
        # Invert the new V using a neat trick: https://math.stackexchange.com/questions/17776/inverse-of-the-sum-of-matrices
        self.VInv = self.VInv - (self.VInv @ B @ self.VInv)/(1+np.trace(B @ self.VInv))
        
        self.rewardsTimesArms += _arm * outcome
        
        # Compute new estimate of theta if we have played all of the arms once
        thetaHat = self.estimate()
        self.estimates[self.t-1] = thetaHat
        
        # Update the state of the bandit
        self.history[self.t-1] = arm
        self.num_plays[arm] += 1
        self.rewards[self.t-1] = outcome
        self.regret[self.t-1] = self.opt_reward - np.dot(_arm, self.theta)
        
        
        # Increment the round
        self.t += 1
        if self.t%1000 == 0:
            print(self.t)
        
    def play(self):
        """
        Play the bandit using LinUCB algorithm

        Returns
        -------
        None.

        """
        
        while self.t <= self.horizon:
            
            arm = self.chooseArm()
            reward = self.pull(arm)
            self.update(arm,reward)
        
        
if __name__=='__main__':
    
    np.random.seed(1234)
    
    d = 100
    n = 100
    T = 10000
    theta = np.zeros((d,))
    theta[0] = 1
    
    # Draw points on unit sphere
    X = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=n)
    X = X/np.linalg.norm(X,ord=2,axis=1)[:,None]
    
    bandit = LinThompsonBandit(arms=X, horizon=T, theta=theta)
    bandit.play()