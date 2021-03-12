#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:05:21 2021

@author: nick
"""

import numpy as np
import mnistLoader as mn

class LinUCBBandit:
    
    def __init__(self,arms,horizon,lambd_reg,contextPool,labelPool,sigma=1,L=1,m2=1):
        """
        Implementation of the linear UCB algorithm for contextual bandits

        Parameters
        ----------
        arms : TYPE
            DESCRIPTION.
        horizon : TYPE
            DESCRIPTION.
        lambd_reg : TYPE
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

        self.L = L
        self.m2 = m2
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
    
        # Store the number of times each arms has been played here
        self.num_plays = np.zeros((self.n,))
        
        # Store the rewards here
        self.rewards = np.zeros((self.horizon,))
        
        # Store our estimates of theta here
        self.estimates = np.zeros((self.horizon,self.d))
        
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
    
    def chooseArm(self,c):
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

        repC = np.repeat(c[None,:],self.n,axis=0)
        feat = np.reshape(repC[:,None,:] * np.eye(self.n)[:,:,None],
                          (self.n,self.d))

        vInvNorm = np.einsum('...i,...i->...', feat.dot(vInv), feat)
        
        # Compute norm w.r.t. vInv using trace trick
        # b = self.arms[:,:,None] * self.arms[:,None,:]
        # B = np.linalg.solve(self.V,b)
        # vInvNorm = np.trace(B,axis1=1,axis2=2)
        
        # Choose arm which maximizes the objective function given in equation 19.13 of Szepsvari/Lattimore
        objFn = np.sum(feat * thetaHat, axis=1) + rootBeta * vInvNorm
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
        
        # Compute new estimate of theta
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
        
        # Now play optimal arm based on maximizing from confidence set
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
    
    bandit = LinUCBBandit(arms=X, horizon=T, lambd_reg=1,
                          contextPool=contextPool, labelPool=labelPool)
    bandit.play()