#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:21:40 2021

@author: nick
"""

import numpy as np
import scipy.linalg as la
import mnistLoader as mn

class FTLBandit:
    
    def __init__(self,arms,horizon,tau,contextPool,labelPool,sigma=1):
        """
        Implementation of the Follow-the-leader algorithm for
        contextual bandits.

        Parameters
        ----------
        arms : TYPE
            DESCRIPTION.
        horizon : TYPE
            DESCRIPTION.
        tau : TYPE
            Number of rounds to pull uniformly at random.
        contextPool : np array
            Pool from which contexts can be drawn.
        sigma : float, optional
            Variance of Gaussian noise added to rewards. The default is 1.

        Returns
        -------
        None.

        """
        
        self.arms = arms
        self.horizon  = horizon

        self.sigma = sigma
        
        # Pool from which contexts can be drawn.
        self.contextPool = contextPool
        self.labelPool = labelPool
        # Dimension of the context space
        self.d = self.contextPool.shape[1]
        
        # Store the contexts and labels observed here
        # self.C = np.zeros((self.horizon,self.contextPool.shape[1]))
        cI = np.random.choice(range(self.contextPool.shape[0]),size=self.horizon)
        self.C = self.contextPool[cI]
        self.L = self.labelPool[cI]
        
        # Number of possible actions (arms)
        self.n = self.arms.shape[0]
        
        # Number of rounds to pull uniformly at random.
        self.tau = tau
           
        # Current round
        self.t = 1
    
        # Store the number of times each arms has been played here
        self.num_plays = np.zeros((self.n,))
        
        # Store the rewards here
        self.rewards = np.zeros((self.horizon,))
        
        # Store feature maps for each context here
        self.feat = np.zeros((self.horizon,self.d*self.n))
        
        # Store our estimate of theta here
        self.estimate = np.zeros((self.n * self.d,))
        
        # Record which arm is pulled in each round here
        self.history = np.zeros(shape=(self.horizon,)).astype(int)
        
        # Record regret at each round here
        self.regret = np.zeros(shape=(self.horizon,))
        
        # Compute the maximum possible reward (for computing regret)
        self.opt_reward = 1
    
    def getContextAndLabel(self):
        
        cI = np.random.choice(range(self.contextPool.shape[0]),size=self.horizon)
        cI = self.t-1
        return self.contextPool[cI],self.labelPool[cI]
    
    def pull(self,arm,context,label):
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
    
    def thetaLoss(self,theta,feat,r):
        """
        Objective function to be minimized for estimating theta

        Parameters
        ----------
        theta : np array
            Guess of what theta should be.
        feat : np array
            Feature map of the contexts/actions.
        r : np array
            Rewards obtained at each round.


        """
        
        return np.sum((r-np.sum(feat*theta,keepdims=True))**2)
        
    def estimateTheta(self):
        """
        Estimate theta using data from first tau pulls.

        Returns
        -------
        thetaHat : float
            The estimator of theta

        """
        
        # Compute feature map
        # feat = np.reshape(self.C[:,None,:] * sbv[:,:,None],
        #                   (self.C.shape[0],self.C.shape[1]*sbv.shape[1]))
        feat = self.feat[:self.t-1]
        b = feat.T @ self.rewards[:self.t-1,None]
        A = feat.T @ feat
        thetaHat = np.linalg.solve(A,b)
        # AInv = la.pinvh(feat.T @ feat)
        # thetaHat = AInv @ b
        
        self.estimate = thetaHat.squeeze()
    
    def chooseArm(self,c):
        """
        Choose the best arm to play by maximizing reward we expect based on
        estimate of theta.

        Parameters
        ----------
        c : np.array
            Observed context.

        Returns
        -------
        arm : int
            Index of the best arm to play in current round

        """
        
        # Compute feature map for each possible action
        repC = np.repeat(c[None,:],self.n,axis=0)
        feat = np.reshape(repC[:,None,:] * np.eye(self.n)[:,:,None],
                          (self.n,c.shape[0]*self.n))
        
        # Choose arm which maximizes the estimated reward
        estReward = np.sum(feat * self.estimate, axis=1)
        optArm = np.argmax(estReward)
        
        return optArm
    
    def update(self,context,arm,reward):
        """
        Update the state of the bandit after a round.

        Parameters
        ----------
        arm : int
            Index of the arm that was played.
        reward : float
            The reward which was observed.


        Returns
        -------
        None.

        """
        
        # Update the state of the bandit
        self.history[self.t-1] = arm
        self.feat[self.t-1] = self.getFeature(context, arm)
        self.num_plays[arm] += 1
        self.rewards[self.t-1] = reward
        self.regret[self.t-1] = self.opt_reward - reward
        
        # Increment the round
        self.t += 1
        if self.t%1000 == 0:
            print(self.t)
            
        if self.t > self.tau:
            self.estimateTheta()
        
    def getFeature(self,context,arm):
        
        return (context[:,None] * np.eye(self.n)[arm,:][None,:]).T.reshape(self.d*self.n)
    
    def play(self):
        """
        Play the bandit using LinUCB algorithm

        Returns
        -------
        None.

        """
        
        # First play each arm once 
        while self.t <= self.tau:
            
            # context,label = self.getContextAndLabel()
            context,label = self.C[self.t-1],self.L[self.t-1]
            
            arm = np.random.choice(range(self.n))
            reward = self.pull(arm,context,label)
            self.update(context,arm,reward)
        
        # Now play optimal arm based on maximizing from confidence set
        while self.t <= self.horizon:
            
            # context,label = self.getContextAndLabel()
            context,label = self.C[self.t-1],self.L[self.t-1]
            
            arm = self.chooseArm(context)
            reward = self.pull(arm,context,label)
            self.update(context,arm,reward)
            
        
        
if __name__=='__main__':
    
    np.random.seed(1234)
    
    d = 100
    n = 100
    T = 50000
    theta = np.zeros((d,))
    theta[0] = 1
    
    # X = np.array(range(10))[:,None]
    # contextPool = np.random.normal(0,1,(1000,5))
    # labelPool = np.random.choice([0,1],size=(1000,1))
    X_train,y_train,X_test,y_test = mn.loadMNIST()
    contextPool = mn.rescale(mn.getRepresentation(X_train,d=16))
    labelPool = y_train
    X = np.unique(y_train)
    
    tau=5000
    
    bandit = FTLBandit(arms=X, horizon=T, theta=theta,
                       tau=tau, contextPool=contextPool, labelPool=labelPool)
    bandit.play()
    