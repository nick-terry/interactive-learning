#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:05:21 2021

@author: nick
"""

import numpy as np

class LinUCBBandit:
    
    def __init__(self,arms,horizon,lambd_reg,theta,sigma=1,L=1,m2=1):
        """
        Implementation of the linear UCB algorithm for Multi-armed bandits

        Parameters
        ----------
        arms : TYPE
            DESCRIPTION.
        horizon : TYPE
            DESCRIPTION.
        lambd_reg : TYPE
            DESCRIPTION.
        theta : TYPE
            DESCRIPTION.
        sigma : float, optional
            Variance of Gaussian noise added to rewards. The default is 1.
        L : float,optional
            Upper bound on norm of action vectors. The default is 1.
        m2 : float, optional
            Upper bound on norm of theta. The default is 1.

        Returns
        -------
        None.

        """
        
        self.arms = arms
        self.horizon  = horizon
        self.lambd_reg = lambd_reg
        # True theta used to generate rewards
        self.theta = theta
        self.L = L
        self.m2 = m2
        self.sigma = sigma
        
        # Dimension of the action space
        self.d = self.arms.shape[1]
        # Number of possible actions (arms)
        self.n = self.arms.shape[0]
        # V matrix used to compute estimate of theta
        self.V = lambd_reg * np.eye(self.d)
        self.VInv = np.linalg.inv(self.V)
        
        # Keep running sum of rewards times arms
        self.rewardsTimesArms = np.zeros((self.d,))
        
        # Current round
        self.t = 1
    
        # Store the number of times each arms has been played here
        self.num_plays = np.zeros((self.n,))
        
        # Store the rewards here
        self.rewards = np.zeros((self.horizon,))
        
        # Store our estimates of theta here
        self.estimates = np.zeros((self.horizon,self.d * self.n))
        
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
        thetaHat = np.linalg.solve(self.V,self.rewardsTimesArms)
        
        return thetaHat
    
    def getRootBeta(self):
        """
        Compute the current Beta term as given in equation 19.8 of Szepsvari/Lattimore

        Returns
        -------
        float
            sqrt(Beta_t)

        """
        return np.sqrt(self.lambd_reg) * self.m2 + \
            np.sqrt(2*np.log(self.n)+self.d * np.log((self.d*self.lambd_reg + self.n*self.L**2)/self.d/self.lambd_reg))
    
    def chooseArm(self):
        """
        Choose the best arm to play according to method in equation 19.13 in Szepsvari/Lattimore

        Returns
        -------
        arm : int
            Index of the best arm to play in current round

        """
        
        rootBeta = self.getRootBeta()
        
        vInv = self.VInv
        thetaHat = self.estimates[self.t-2]
        # vInvNorm = np.sum((self.arms @ vInv[None,:,:]).squeeze() * self.arms,axis=1)
        vInvNorm = np.einsum('...i,...i->...', self.arms.dot(vInv), self.arms)
        
        # Compute norm w.r.t. vInv using trace trick
        # b = self.arms[:,:,None] * self.arms[:,None,:]
        # B = np.linalg.solve(self.V,b)
        # vInvNorm = np.trace(B,axis1=1,axis2=2)
        
        # Choose arm which maximizes the objective function given in equation 19.13 of Szepsvari/Lattimore
        objFn = np.sum(self.arms * thetaHat, axis=1) + rootBeta * vInvNorm
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
        self.rewardsTimesArms += _arm * outcome
        
        # Update V matrix and its inverse
        B = np.outer(_arm,_arm)
        self.V += B
        # Invert the new V using a neat trick: https://math.stackexchange.com/questions/17776/inverse-of-the-sum-of-matrices
        self.VInv = self.VInv - (self.VInv @ B @ self.VInv)/(1+np.trace(B @ self.VInv))
        
        # Compute new estimate of theta if we have played all of the arms once
        if self.t >= self.n:
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
        
        # First play each arm once 
        while self.t <= self.n:
            
            arm = self.t - 1
            reward = self.pull(arm)
            self.update(arm,reward)
        
        # Now play optimal arm based on maximizing from confidence set
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
    
    bandit = LinUCBBandit(arms=X, horizon=T, lambd_reg=.2, theta=theta)
    bandit.play()