#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:05:21 2021

@author: nick
"""

import numpy as np
import torch as t
import matplotlib.pyplot as plt

class TransductiveBandit:
    
    def __init__(self,arms,Z,zHat,initAlloc,horizon,lambd_reg,theta,N=100,eta=.001,sigma=1,L=1,m2=1):
        """
        Implementation of a linear transductive bandit algorithm based on iteratively 
        refining a deterministic sampling allocation

        Parameters
        ----------
        arms : TYPE
            DESCRIPTION.
        Z : np array
            Choices (vectors) from which we are trying to maximize z^T \theta
        zHat : np array
            Guess of which vector in Z is optimal
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
        L : float,optional
            Upper bound on norm of action vectors. The default is 1.
        m2 : float, optional
            Upper bound on norm of theta. The default is 1.

        Returns
        -------
        None.

        """
        
        self.arms = arms
        self.Z = Z
        self.zHat = zHat
        self.horizon  = horizon
        self.allocation = np.zeros((self.horizon+1,self.arms.shape[0]))
        self.allocation[0,:] = initAlloc
        self.N = N
        
        self.lambd_reg = lambd_reg
        # True theta used to generate rewards
        self.theta = theta
        self.eta = eta
        self.L = L
        self.m2 = m2
        self.sigma = sigma
        
        # Dimension of the action space
        self.d = self.arms.shape[1]
        # Number of possible actions (arms)
        self.n = self.arms.shape[0]
        # V matrix used to compute estimate of theta
        self.V = np.zeros((self.horizon,self.d,self.d))
        
        # Keep running sum of rewards times arms
        self.sumRewardsTimesArms = np.zeros((self.horizon,self.d))
        
        # Current round
        self.t = 1
    
        # Store the number of times each arms has been played here
        self.num_plays = np.zeros((self.n,))
        
        # Store the rewards here
        self.rewards = np.zeros((self.horizon,self.N))
        
        # Store our estimates of theta here
        self.estimates = np.zeros((self.horizon,self.d))
        
        # Store z_t here
        self.zMax = np.zeros((self.horizon,))
        
        # Record which arm is pulled in each round here
        self.history = np.zeros(shape=(self.horizon,self.N)).astype(int)
        
    def drawArms(self,allocation):
        
        sampledArms = np.random.choice(range(self.n),self.N,p=allocation)
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
        outcome = np.dot(action,self.theta) + np.random.normal(0,self.sigma**2,size=self.N)
        
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
        thetaHat = np.linalg.solve(self.V[self.t-1] + self.lambd_reg * np.eye(self.d),
                                   self.sumRewardsTimesArms[self.t-1])
        
        return thetaHat
    
    
    def updateAllocation(self,allocation,arm,z,theta):
        
        # _arm = t.tensor(self.arms[arm])
        _arm = t.tensor(self.arms)
        allocation = t.tensor(allocation,requires_grad=True)
        
        A_lambda_inv = t.inverse(t.sum(allocation[:,None,None] * (_arm[:,None,:] * _arm[:,:,None]),axis=0))
        
        z = t.tensor(self.arms[z])
        zHat = t.tensor(self.arms[self.zHat])
        theta = t.tensor(theta)
        diff = z - zHat 
        
        objFn = (diff.T @ A_lambda_inv @ diff)/t.sum(diff * theta)**2
        
        # Use pytorch autograd to compute gradient of this expression w.r.t the allocation
        objFn.backward()
        
        # Take gradient step
        newAllocation = allocation - self.eta * allocation.grad
        
        # Project back to simplex
        newAllocation = newAllocation/t.sum(newAllocation)
        
        if t.any(t.isnan(newAllocation)) or t.any(newAllocation<0):
            print('oops')

        return np.array(newAllocation.detach())
    
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
        self.sumRewardsTimesArms[self.t-1] = np.sum(_arm * outcome[:,None],axis=0)
        
        # Update V matrix and its inverse
        self.V[self.t-1] = np.sum(_arm[:,None,:] * _arm[:,:,None],axis=0)
        
        # Compute least squares estimator of theta
        thetaHat = self.estimate()
        self.estimates[self.t-1] = thetaHat
        
        # Pick best z in Z based on new estimate of theta
        zMax = np.argmax(np.sum(self.Z * thetaHat,axis=1))
   
        # If these don't agree, update allocation
        if zMax != self.zHat:
            
            newAllocation = self.updateAllocation(self.allocation[self.t-1],arm,zMax,thetaHat)
            self.allocation[self.t] = newAllocation
        
        else:
            
            self.allocation[self.t] = self.allocation[self.t-1]
            
        # Update the state of the bandit
        self.history[self.t-1,:] = arm
        self.num_plays[arm] += 1
        self.rewards[self.t-1] = outcome
        self.zMax[self.t-1] = zMax
        
        # Increment the round
        self.t += 1
        if self.t%1000 == 0:
            print('Done with round {}'.format(self.t))
        
    def play(self):
        """
        Play the bandit using LinUCB algorithm

        Returns
        -------
        None.

        """
        
        while self.t <= self.horizon:
            
            arms = self.drawArms(self.allocation[self.t-1])
            rewards = self.pull(arms)
            self.update(arms,rewards)
        
        
if __name__=='__main__':
    
    np.random.seed(1234)
    
    d = 25
    n = d
    T = 10000
    
    theta = np.zeros((d,))
    # theta[1] = 1
    theta = np.random.normal(0,9,(d,))
    
    Z = np.eye(d)
    # Index of the optimal choice in Z
    zHat = 1
    initAlloc = np.ones((n,))/n
    
    # # Draw points on unit sphere
    # X = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=n)
    # X = X/np.linalg.norm(X,ord=2,axis=1)[:,None]
    X = np.eye(d)
    
    bandit = TransductiveBandit(arms=X, Z=Z, zHat=zHat, initAlloc=initAlloc, horizon=T, lambd_reg=.2, theta=theta,
                                N=100,eta=1e-3)
    bandit.play()
    
    print()
    print('How many times did we select each z as the best?')
    for i in range(Z.shape[0]):
        print('e_{}: {}'.format(i+1,np.sum(bandit.zMax==i)))
    
    bestArm = np.argmax(np.sum(bandit.Z * theta,axis=1))
    correctRate = np.cumsum((bandit.zMax==bestArm))/np.arange(1,T+1,1)
    
    fig,ax = plt.subplots(1)
    ax.plot(correctRate[1:])
    ax.set_xlabel('Round (t)')
    ax.set_ylabel('Cumulative Correct Best Arm Rate')
    
    fig,ax = plt.subplots(1)
    for i in range(d):
        ax.plot(bandit.allocation[:,i],label='Arm {}'.format(i+1),alpha=.75)
    
    ax.set_xlabel('Round (t)')
    ax.set_ylabel('Allocation Mass')
    ax.legend()
    
    # fig,axes = plt.subplots(d)
    # for i in range(d):
    #     ax = axes[i]
    #     ax.plot(bandit.allocation[:,i],label='Arm {}'.format(i))
    
    #     ax.set_xlabel('Round (t)')
    #     ax.set_ylabel('Allocation Mass')
        
    #     if i!=bestArm:
    #         ax.set_title('Arm {}'.format(i))
    #     else:
    #         ax.set_title('Arm {} (Best Arm)'.format(i))