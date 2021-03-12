#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:43:32 2021

@author: nick
"""

import numpy as np
import matplotlib.pyplot as plt

'''
Create a bandit problem to play with the explore-then-commit method. Rewards are 
normally distributed where each arm has equal variance.
'''
class Bandit():
    
    '''
    Create the bandit
    
    Parameters:
        
        arms: a numpy array containing the true mean for each arm
        
        horizon: the number of rounds in which we will make a decision
        
        m: determines how many times pull pull each arm before committing
    '''
    def __init__(self,arms,horizon=10,m=1):
        
        #Store the true mean of each arm here.
        self.arms = arms
        #The largest mean reward of all arms
        self.mu_max = np.max(self.arms)
        self.horizon = horizon
        self.m = m
        
        #At a minimum, we should have a larger horizon than the number of arms.
        assert(self.horizon>self.numArms)
        
        #Store the number of times each arms has been played here
        self.N = np.zeros_like(self.arms)
        
        #Store our estimates of the mean of each arm here
        self.estimates = np.zeros_like(self.arms)
        self.estimate_error = np.zeros((self.horizon,self.numArms))
        
        #Record which arm is pulled in each round here
        self.history = np.zeros(shape=(self.horizon,))
        
        #Record regret at each round here
        self.regret = np.zeros(shape=(self.horizon,))
        
    '''
    Pull an arm of the bandit
    
    Parameters:
        
        arm: the index i in {1,...,N} of the arm to pull
        
    Returns: the reward generated from N(mu_i,1)
    '''
    def _pull(self,arm):
        outcome = np.random.normal(self.arms[arm-1],1)
        return outcome
    
    '''
    Update our estimate of the arm's mean and the number of plays
    '''
    def update(self,arm,outcome):
        self.estimates[arm-1] = (self.estimates[arm-1]*self.N[arm-1] + outcome)/(self.N[arm-1]+1)
        self.N[arm-1] += 1
    
    '''
    Choose an arm to play using explore-then-commit method.
    '''
    def play(self):
        t = 1        
        while t <= self.numArms*self.m:
           
            #Play each arm  m times to get initial estimates of success prob
            arm = (t % self.numArms) + 1
            self.pull(t,arm)
            
            t += 1

        #Choose I_t to be the arm with largest sample mean
        arm = np.argmax(self.estimates)+1
        
        #Play the chosen arm until time horizon
        while t <= self.horizon:
            
            self.pull(t,arm)
        
            t += 1
 
    '''
    Play the given arm, then update some relevant states
    '''
    def pull(self,t,arm):
        outcome = self._pull(arm)
        self.update(arm,outcome)
        self.estimate_error[t-1,:] = np.abs(self.estimates-self.arms)
        self.history[t-1] = arm
        if t > 1:
            self.regret[t-1] = self.regret[t-2] + self.mu_max - self.arms[arm-1]
        else:
            self.regret[t-1] = self.mu_max - self.arms[arm-1]
    
    @property
    def numArms(self):
        return self.arms.shape[0]
        
    @property
    def randomArmIndex(self):
        return np.random.choice(range(1,self.numArms+1))
    
'''
Create a bandit problem to play with the UCB method
'''    
class UCBBandit(Bandit):
    
    def __init__(self, arms, horizon=10):
        super(UCBBandit,self).__init__(arms, horizon)
    
    '''
    Choose an arm to play using UCB method
    '''
    def play(self):
        t = 1   
        
        #Play each arm once to get initial estimates of success prob
        while t <= self.numArms:
           
            arm = t
            self.pull(t,arm)
            
            t += 1
            
        #After playing each arm once, play by choosing the largest upper bound each round
        while t <= self.horizon:
            
            #Compute confidence interval half-width
            hw = np.sqrt(2*np.log(2*self.numArms*self.horizon**2)/self.N)
            
            #Choose arm with highest upper bound (sample mean + CI half-width)
            arm = np.argmax(self.estimates + hw) + 1
            
            #Play the chosen arm, then update the estimate and history
            self.pull(t,arm)
            
            t += 1
            
'''
Create a bandit problem to play with the Thompson sampling method
'''        
class ThompsonBandit(Bandit):
    
    def __init__(self, arms, horizon=10):
        super().__init__(arms, horizon)
        
        #Take the prior to be N(0,1)
        #We will store the posterior mean and variance for each arm here
        self.params = np.zeros(shape=(self.numArms,2))
        self.params[:,1] = 1
        
        #Initial estimates are drawn from prior (Normal) distribution
        self.estimates = np.zeros_like(self.arms)
        for armIndex in range(self.numArms):
            self.estimates[armIndex] = np.random.normal(self.params[armIndex,0],self.params[armIndex,1])
        
    '''
    Compute posterior and update and the number of plays
    '''
    def update(self,arm,outcome):
        
        self.N[arm-1] += 1
        
        #Compute new posterior mean
        self.params[arm-1,0] = (self.params[arm-1,0]*self.N[arm-1] + outcome) / (self.N[arm-1]+1)
        # self.params[arm-1,0] = (self.params[arm-1,0] + outcome)/2
        
        #Compute new posterior variance
        self.params[arm-1,1] = 1/(1+self.N[arm-1])
        
        #Draw new estimates from posterior
        for armIndex in range(self.numArms):
            self.estimates[armIndex] = np.random.normal(self.params[armIndex,0],self.params[armIndex,1])
              
    '''
    Always choose arm with largest estimate drawn from posterior.
    '''
    def play(self):
        t = 1        
        while t <= self.horizon:

            #Choose I_t to be the arm with largest draw
            arm = np.argmax(self.estimates)+1
            
            self.pull(t,arm)
        
            t += 1
        
# Perform experiment for problem 4.1
def partA():
    
    np.random.seed(4321)
    
    T = 12500
    n = 10
    mu = np.array([.1,]+[0,]*(n-1))
    mVals = (5,25,100)
    
    #Run explore then commit for several values of m
    etcBandits = []
    for m in mVals:
        bandit = Bandit(mu,T,m)
        bandit.play()
        etcBandits.append(bandit)
        
    #Run UCB
    ucbBandit = UCBBandit(mu,T)
    ucbBandit.play()
    
    #Run Thompson Sampling
    tsBandit = ThompsonBandit(mu,T)
    tsBandit.play()
    
    #Plot the regret of each bandit
    fig,ax = plt.subplots(1)
    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Regret')
    # ax.set_ylim([0,.1])

    x = range(T)
    for b,m in zip(etcBandits,mVals):
        ax.plot(x,b.regret,label='ETC w/ m={}'.format(m))
    
    ax.plot(x,ucbBandit.regret,label='UCB')
    ax.plot(x,tsBandit.regret,label='Thompson')
    plt.legend()
    
    return etcBandits,ucbBandit,tsBandit

# Perform experiment for problem 4.2
def partB():
    
    np.random.seed(1234)
    
    T = 50000
    n = 40
    mu = np.array([1,]+[1-1/np.sqrt(i-1) for i in range(2,n+1)])
    mVals = (5,25,100)
    
    #Run explore then commit for several values of m
    etcBandits = []
    for m in mVals:
        bandit = Bandit(mu,T,m)
        bandit.play()
        etcBandits.append(bandit)
        
    #Run UCB
    ucbBandit = UCBBandit(mu,T)
    ucbBandit.play()
    
    #Run Thompson Sampling
    tsBandit = ThompsonBandit(mu,T)
    tsBandit.play()
    
    #Plot the regret of each bandit
    fig,ax = plt.subplots(1)
    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Regret')
    # ax.set_ylim([0,.1])

    x = range(T)
    for b,m in zip(etcBandits,mVals):
        ax.plot(x,b.regret,label='ETC w/ m={}'.format(m))
    
    ax.plot(x,ucbBandit.regret,label='UCB')
    ax.plot(x,tsBandit.regret,label='Thompson')
    plt.legend()
    
    return etcBandits,ucbBandit,tsBandit

etcBandits,ucbBandit,tsBandit = partA()
# etcBandits,ucbBandit,tsBandit = partB()