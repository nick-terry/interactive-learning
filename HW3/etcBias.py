#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:21:40 2021

@author: nick
"""

import numpy as np
import scipy.optimize as opt
import sklearn.linear_model as lm
import mnistLoader as mn

class ETCBiasBandit:
    
    def __init__(self,arms,horizon,tau,contextPool,labelPool,sigma=1):
        """
        Implementation of the Explore-then-commit algorithm for
        contextual bandits using "model the bias" approach

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
        
        # Store the contexts observed here
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
        self.rewards = np.zeros((self.horizon,)).astype(int)
        
        # Store our estimate of theta here
        self.estimate = np.zeros((self.n * self.d,))
        
        # Record which arm is pulled in each round here
        self.history = np.zeros(shape=(self.horizon,)).astype(int)
        
        # Record regret at each round here
        self.regret = np.zeros(shape=(self.horizon,))
        
        # Compute the maximum possible reward (for computing regret)
        self.opt_reward = 1
    
    def getContextAndLabel(self):
        
        cI = np.random.choice(range(self.contextPool.shape[0]))
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

        
    def buildClassifier(self):
        
        classifier = lm.LogisticRegression()
        
        dataMask = self.rewards[:self.t]==1
        
        # Choose training data to be times where reward==1
        X = self.C[:self.t][dataMask]
        y = self.history[:self.t][dataMask].reshape(-1,1)
        
        classifier.fit(X,y)
        
        self.classifier = classifier
    
    def chooseArm(self,c):
        """
        Choose the best arm using classifier

        Parameters
        ----------
        c : np.array
            Observed context.

        Returns
        -------
        arm : int
            Index of the best arm to play in current round

        """
        
        optArm = self.classifier.predict(c[None,:])
        
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
        self.C[self.t-1] = context
        self.history[self.t-1] = arm
        self.num_plays[arm] += 1
        self.rewards[self.t-1] = reward
        self.regret[self.t-1] = self.opt_reward - reward
        
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
        while self.t <= self.tau:
            
            context,label = self.C[self.t-1],self.L[self.t-1]
            
            arm = np.random.choice(range(self.n))
            reward = self.pull(arm,context,label)
            self.update(context,arm,reward)
        
        self.buildClassifier()
        
        # Now play optimal arm based on maximizing from confidence set
        while self.t <= self.horizon:
            
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
    
    tau=15000
    
    bandit = ETCBiasBandit(arms=X, horizon=T, theta=theta,
                       tau=tau, contextPool=contextPool, labelPool=labelPool)
    bandit.play()
    