#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 09:13:15 2021

@author: nick
"""

import numpy as np
import matplotlib.pyplot as plt

class GOptElimBandit:
    
    def __init__(self,arms,horizon,theta,sigma=1):
        """
        Implements a elimination-style algorithm with G-optimal design
        for multi-armed bandit problems

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
            
        Returns
        -------
        None.

        """
        self.array_scale_factor = 10
        
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
        self.V = []
        # Current round
        self.t = 1
        # Current phase
        self.l = 1
        # Track the starting time of each phase and ending time of each phase
        self.tl = np.zeros((self.array_scale_factor*self.horizon,2)).astype(int)
        self.tl[0,0] = 1
        # Current G-optimal design. Use list bc we don't know how many times we will do this
        self.pi = []
        # Number of times each arm is pulled in each phase
        self.pullList = []
        
        # The indices of arms which are not yet eliminated
        self.actionSet = list(range(self.n))
    
        # Store the number of times each arms has been played here
        self.num_plays = np.zeros((self.n,))
        
        # Store the rewards here
        self.rewards = np.zeros((self.array_scale_factor*self.horizon,))
        
        # Store our estimates of theta here
        self.estimates = []
        # self.estimate_error = np.zeros((self.horizon,self.numArms))
        
        # Record which arm is pulled in each round here
        self.history = np.zeros(shape=(self.array_scale_factor*self.horizon,)).astype(int)
        
        # Record regret at each round here
        self.regret = np.zeros(shape=(self.array_scale_factor*self.horizon,))
        
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
    
    def estimate(self,pi,T_l):
        """
        Compute the regularized least squares estimator for theta. This should
        happen when self.t is up-to-date.

        Parameters
        ----------
        pi : TYPE
            DESCRIPTION.
        Tl : TYPE
            DESCRIPTION.

        Returns
        -------
        thetaHat : float
            The regularized least squares estimator of theta in stage l

        """
        
        # Get starting and stopping time of current phase l
        tl,plusTl = self.tl[self.l-1,0],self.tl[self.l-1,1]
        
        b = np.sum(self.rewards[tl-1:plusTl-1,None]*self.arms[self.history[tl-1:plusTl-1]],axis=0)
        
        remainingArms = self.arms[np.array(self.actionSet)]
        V = np.sum(T_l[:,None,None] * (remainingArms[:,:,None]*remainingArms[:,None]),axis=0)
        self.V.append(V)
        
        VInv = np.linalg.pinv(V)
        thetaHat = VInv @  b
        # thetaHat = np.linalg.solve(V,b)
        
        
        return thetaHat
    
    def eliminate(self,thetaHat):
        """
        Eliminate arms from action set using the estimate thetaHat

        Parameters
        ----------
        thetaHat : np array
            Estimate of theta

        Returns
        -------
        None.

        """
        
        remainingArms = self.arms[np.array(self.actionSet)]
        
        rewards = np.sum(remainingArms * thetaHat, axis=1)
        maxReward = np.max(rewards)
        
        newActionSetMask = (maxReward - rewards) <= 2**(1-self.l)
        
        self.actionSet = list(np.array(self.actionSet)[newActionSetMask])
    
    def getGOpt(self,arms,N=1000):
        """
        Compute a G-optimal design vector for arms.

        Parameters
        ----------
        arms : list
            List of indices of the arms for which we will generate the G-Optimal design.

        Returns
        -------
        pi : np array
            A probability distribution over the arms.

        """
        
        # Get the actual arms from indices
        _arms = self.arms[np.array(arms)]
        
        _,_,pi = frankWolfe(_arms, N)
        
        return pi
    
    def getT_l(self,pi):
        """
        Compute the number of times to pull each arm in phase l

        Returns
        -------
        np array
            sqrt(Beta_t)

        """
        
        e_l = 2**-self.l
        
        T_l = np.ceil(2*self.d*pi/e_l**2 * np.log(len(self.actionSet)*self.l*(self.l+1)*self.n))
        
        # Set the ending time for phase l
        self.tl[self.l-1,1] = (self.tl[self.l-1,0] + np.sum(T_l)).astype(int)
        
        return T_l.astype(int)
    
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
        
        # Update the state of the bandit
        self.history[self.t-1] = arm
        self.num_plays[arm] += 1
        self.rewards[self.t-1] = outcome
        self.regret[self.t-1] = self.opt_reward - np.dot(_arm, self.theta)
            
        # Increment the round
        self.t += 1
        if self.t%1000 == 0:
            print(self.t)
        
    def endPhase(self):
        """
        Perform estimation of theta and elimination for the end of the phase.

        Returns
        -------
        None.

        """
        
        pi = self.pi[self.l-1]
        T_l = self.pullList[self.l-1]
        
        # Estimate theta
        thetaHat = self.estimate(pi,T_l)
        self.estimates.append(thetaHat)
        
        # Eliminate arms
        self.eliminate(thetaHat)
        
        # Increment phase and record start time of new phase
        self.l += 1
        self.tl[self.l-1,0] = self.t
        
    
    def play(self):
        """
        Play the bandit using G-optimal design and elimination.

        Returns
        -------
        None.

        """
        
        while self.t <= self.horizon:
            
            # Compute the G-optimal design for current action set
            pi = self.getGOpt(self.actionSet)
            self.pi.append(pi)
            
            # Compute number of times to play each arm this phase
            T_l = self.getT_l(pi)
            self.pullList.append(T_l)
            
            plusTl = self.tl[self.l-1,1]
            # For the remainder of phase l, pull the arms we determined w/ above computations
            while self.t < plusTl:
                
                for arm,numPulls in zip(self.actionSet,T_l):
                    for i in range(numPulls):
                        
                        # Pull arm and get reward
                        outcome = self.pull(arm)
                        
                        # Update state of the bandit
                        self.update(arm,outcome)
                
            # End the phase and eliminate arms
            self.endPhase()
        
        # Truncate oversized arrays
        self.regret = self.regret[:self.t]
        self.history = self.history[:self.t]
        self.rewards = self.rewards[:self.t]
        self.tl = self.tl[:self.t]
        
def f(X,lambd,returnAll=False):
    """
    Objective function for G-optimal design

    Parameters
    ----------
    X : np array
        The design points.
    lambd : np array
        The probability vector which gives the G-optimal sampling distribution.
    returnAll : boolean, optional
        Whether or not to return some intermediate values from computing f.
        The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    
     # Create matrix A(lambda)
    A_lambd = np.sum(lambd[:,None,None] * (X[:,:,None]*X[:,None]),axis=0)
    
    # Compute quadratic form norm of each x_i w.r.t A(lambda)^{-1}
    AInv = np.linalg.pinv(A_lambd)

    Xnorm = np.diag((X @ AInv) @ X.T)
    
    
    if returnAll:
        return np.max(Xnorm), AInv, Xnorm
    else:
        return np.max(Xnorm)

def df(X,lambd):
    """
    Compute the gradient of f(X) w.r.t lambda

    Parameters
    ----------
    X : np array
        Design points. Each row of X is a vector x_i^T.
    lambd : np array
        The probability vector which gives the G-optimal sampling distribution.
    Returns
    -------
    np array

    """
    
    _,AInv,Xnorm = f(X,lambd,returnAll=True)
    
    # Find x_i with largest norm and compute gradient 
    iMax = np.argmax(Xnorm)
    grad = - X[None,iMax,:] @ AInv @ (X[:,:,None] * X[:,None,:]) @ AInv @ X[None,iMax,:].T
    
    return grad.squeeze()

def frankWolfe(X,N):
    """
    Implements the Frank-Wolfe algorithm to find a G-optimal design

    Parameters
    ----------
    X : np array
        Design points.
    N : int
        The number of design points we wish to check (with replacement)

    Returns
    -------
    I : np array
        Indices of the chosen design points
    fArray : np array
        The objective function values at each iteration.

    """
    
    n = X.shape[0]
    d = X.shape[1]
    
    # Choose first 2d entries randomly
    _I = np.random.choice(range(n),size=2*d,replace=True)
    
    # Choose remaining entries greedily
    I = np.zeros((N,)).astype(int)
    I[:2*d] = _I
    
    lambd = np.bincount(_I,minlength=n)/2/d
    fArray = np.zeros((N-2*d,))
    fArray[0] = f(X,lambd)
    
    for t in range(2*d+1,N):
        
        # Find j that minimizes gradient
        grad = df(X,lambd)
        I[t] = np.argmin(grad)
        
        # Update lambd
        eta_t = 2/(t+1)
        lambd = (1-eta_t) * lambd
        lambd[I[t]] += eta_t
        
        fArray[t-(2*d+1)+1] = f(X,lambd)
        
    return I,fArray,lambd

if __name__=='__main__':
    
    np.random.seed(1234)
    
    d = 25
    n = 100
    T = 1000000
    theta = np.zeros((d,))
    theta[0] = 1
    
    # Draw points on unit sphere
    X = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=n)
    X = X/np.linalg.norm(X,ord=2,axis=1)[:,None]
    
    bandit = GOptElimBandit(arms=X, horizon=T, theta=theta)
    bandit.play()
    
    plt.plot(np.cumsum(bandit.regret))