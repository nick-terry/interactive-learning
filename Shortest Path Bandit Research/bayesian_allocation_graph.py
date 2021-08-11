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
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra

from graphArgmax import ShortestPathFinder

class TransductiveBandit:
    
    def __init__(self,arms,nSteps,nRounds,graph,source,target,
                 lambd_reg,theta,delta=.05,eta=1e-3,sigma=1,v=2):
        """
        Implementation of a linear transductive bandit algorithm based on iteratively 
        refining a deterministic sampling allocation

        Parameters
        ----------
        arms : TYPE
            DESCRIPTION.
        Z : np array
            Choices (vectors) from which we are trying to maximize z^T \theta
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
        self.nSteps = nSteps
        self.nRounds = nRounds
        
        self.allocation = np.zeros((self.nRounds+1,self.arms.shape[0]))
        self.allocation[0,:] = np.ones(self.arms.shape[0])/self.arms.shape[0]
        
        self.lambd_reg = lambd_reg
        
        # Create an argmax solver
        self.solver = ShortestPathFinder(graph,source,target,arms)
        
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
        self.posteriorMean = np.zeros((self.nRounds+1,self.d))
        self.posteriorCovar = np.zeros((self.nRounds+1,self.d,self.d))
        self.posteriorCovar[0,:,:] = np.eye(self.d)
        
        # Current round
        self.k = 1
        # Current step
        self.t = 1
        
        # Base for number of samples to take in each round (e.g. self.v**self.k)
        self.v = v
    
        # Store the number of times each z in \mathcal{Z} has been played here
        # self.numTimesOpt = np.zeros((self.Z.shape[0],))
        
        # Store our estimates of theta here
        self.estimates = np.zeros((self.nRounds,self.nSteps,self.d))
        self.armPlays = np.zeros((self.n,))
        
        # Store z_t here
        self.z = -np.ones((self.nRounds,self.nSteps,2*self.n))
        
        self.sampleComplexity = 0
        
        self.armsHistory = []
        self.rewardsHistory = []
        self.B_t = np.eye(self.d)
        self.rewTimesArms = np.zeros((self.d,))
        
        self.zHat = None
    
    def getArgmax(self,theta):
        
        optLength,z = self.solver.getArgmax(theta)
        # z = self.getZIndices(z)
        return z
    
    def getZIndices(self,z):
        if len(z.shape) > 1:
            zI = np.argmax(np.all(self.Z[None,:,:] == z.squeeze().T[:,None,:],axis=-1),axis=-1)
        else:
            zI = np.argmax(np.all(self.Z[None,:,:] == z.squeeze().T,axis=-1),axis=-1)
        return zI
    
    def eta(self):
        return self._eta
    
    def drawArms(self,allocation):
        
        sampledArms = np.random.choice(range(self.n),int(np.ceil(self.v**(self.k+2))),p=allocation)
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
        thetaHat = Binv @ self.rewTimesArms
        
        return thetaHat,Binv
    
    def updateAllocationFW(self,allocation,z1,z2,epochs=1000):
        
        diff = z1 - z2
        diff = np.unique(diff,axis=0)
        
        # Multiply by feature map
        X = self.arms
        diff = diff @ X
        
         # Define some stuff for minimizing inner product over simplex
        A_ub = -np.eye(self.n)
        b_ub = np.zeros((self.n,1))
        
        A_eq = np.ones((self.n,1)).T
        b_eq = 1
        
        for epoch in range(epochs):

            # Compute inverse design matrix
            try:
                A_lambda_inv = np.linalg.inverse(t.sum(allocation[:,None,None] * (X[:,None,:] * X[:,:,None]),axis=0))
            # If the matrix is singular, we can take pseudoinverse instead
            except Exception:
                A_lambda_inv = np.linalg.pinv(np.sum(allocation[:,None,None] * (X[:,None,:] * X[:,:,None]),axis=0))

            yHat = np.argmax(np.sum(diff @ A_lambda_inv  * diff, axis=1))
            
            # Compute gradient            
            grad = (-diff[yHat] @ A_lambda_inv @ (X[:,None,:] * X[:,:,None]) @ A_lambda_inv @ diff[yHat])
            
            # Update using Frank-Wolfe step
            try:
                aMin = linprog(grad,A_ub,b_ub,A_eq,b_eq).x
            except Exception as e:
                print(e)
                
            gamma = 2/(epoch+2)
            _allocation = (1-gamma) * allocation + gamma * aMin

        return allocation
    
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
        theta = np.random.multivariate_normal(self.posteriorMean[self.k-1], self.posteriorCovar[self.k-1], 2*self.nSteps).T
        
        # Pick best z in Z based on draws of theta using argmax solver
        z1 = self.getArgmax(theta[:,:self.nSteps]).T
        z2 = self.getArgmax(theta[:,self.nSteps:]).T
        
        # self.numTimesOpt += np.bincount(z1, minlength=self.Z.shape[0])
        # self.numTimesOpt += np.bincount(z2, minlength=self.Z.shape[0])

        self.z[self.k-1,:,:self.n] = z1
        self.z[self.k-1,:,self.n:] = z2

        newAllocation = self.updateAllocationFW(self.allocation[self.k-1],z1,z2)
        self.allocation[self.k-1] = newAllocation
            
        # Draw arms from current allocation and pull
        arms = self.drawArms(self.allocation[self.k-1])
        rewards = self.pull(arms)
        
        # Track history 
        # self.armsHistory.append(_arms)
        # self.rewardsHistory.append(rewards)
        
        # Update B matrix
        self.updateB_t(arms)
        
        # Update rewards times arms
        self.rewTimesArms += np.sum(self.arms[arms] * rewards[:,None],axis=0)
        
        # Update sample complexity
        self.sampleComplexity += arms.shape[0]
        
        # Compute posterior params
        mu,sigma = self.getPosterior()
            
        self.posteriorCovar[self.k] = sigma
        self.posteriorMean[self.k] = mu
        
        self.allocation[self.k] = newAllocation
        
        # print('Done with round {}'.format(self.k))
        
    def updateB_t(self,arms):
        
        playsPerArm = np.bincount(arms,minlength=self.n)
        
        self.armPlays += playsPerArm
        self.B_t += np.sum(playsPerArm[:,None,None] * (self.arms[:,None,:] * self.arms[:,:,None]),axis=0)
        
    # This is the old terminating condition for when the size of Z is known precisely
    def checkTerminationCond2(self):
        
        zHat = self.getArgmax(self.posteriorMean[self.k])
        diff = np.squeeze(self.Z[zHat][None,:] - self.Z)
        diff = diff[np.sum(np.abs(diff),axis=1)>0]
        
        A_lambda_inv = np.linalg.pinv(self.B_t)
        
        A_inv_norm = np.sum((diff @ A_lambda_inv[None,:,:]).squeeze() * diff,axis=1)
        
        rhs = np.sqrt(2*A_inv_norm*np.log(2*self.Z.shape[0]*self.k**2/self.delta))
        lhs = np.sum(diff * self.posteriorMean[self.k],axis=1)
        
        terminateNow = np.all(lhs >= rhs)
        
        return terminateNow,zHat
    
    def checkTerminationCond(self):
        
        z_t = np.reshape(self.z[self.k-1,:,:],(2*self.nSteps,self.n))
        
        # Check if we only got a single z_t in the previous stage
        terminateNow = np.unique(z_t,axis=0).shape[0] == 1
       
        zHat = z_t[0] if terminateNow else None
        
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

    # Run several different size graphs w/ feature vectors for each edge    

    seed = 123456
    np.random.seed(seed)

    nNodesTup = (10,15,20,25,30)
    nEdgesPerNodeTup = (5,5,10,10,15)
    nReps = 10
    
    sampleComplexity = np.zeros((len(nNodesTup),))
    incorrectCounts = np.zeros((len(nNodesTup),))
    for i in range(len(nNodesTup)):
        
        nNodes = nNodesTup[i]
        nEdgesPerNode = nEdgesPerNodeTup[i]
        
        # define some params for generating a random graph
        
        # out-degree of each node in the graph
        
        alpha = .5
        
        G = nx.random_k_out_graph(nNodes,nEdgesPerNode,alpha,seed=seed)
        
        # total number of edges
        n = nNodes * nEdgesPerNode
        source = 1
        target = 0
        
        # Create the feature vectors
        featureDim = 5
        fX = np.random.multivariate_normal(np.array([1,1,7,3,2]), .5*np.eye(featureDim), size=n)
        fX = np.maximum(fX,np.zeros_like(fX))
        
        # theta = np.random.normal(0,2,size=n)
        theta = np.zeros((featureDim))
        theta[0] = 1
        theta[1] = .9998
        
        weightEst = fX @ theta
        edgeToWeight = dict(zip(G.edges,[{'weight':x} for x in weightEst.tolist()]))    
        nx.set_edge_attributes(G,edgeToWeight)
        length,path = single_source_dijkstra(G,source,target,weight='weight')
        
        zStar = np.zeros_like(weightEst)
        zStar[np.array(path)] = 1
        
        T = 2500
        K = 1000
          
        results = []
        incorrectCnt = 0
        for rep in range(nReps):
            
            print('Replication {} of {}'.format(rep,nReps))
            
            bandit = TransductiveBandit(fX,T,K,G,source,target,lambd_reg=0,theta=-theta,v=2)
            bandit.play() 
            results.append(bandit.sampleComplexity)
            
            # Check that the result is correct
            try:
                assert(np.all(bandit.zHat==zStar))
            except:
                print('Claimed best arm: {}'.format(bandit.zHat))
                print('Best arm: {}'.format(zStar))
        
                incorrectCnt += 1
                
        sampleComplexity[i] = sum(results) 
        incorrectCounts[i] = incorrectCnt
    
    return sampleComplexity/nReps,incorrectCounts/nReps


def runBenchmark2():

    # Run several different size graphs w/ feature vectors for each edge    
    # this is a harder estimation problem than the first benchmark

    seed = 123456
    np.random.seed(seed)

    nNodesTup = [2**i for i in range(3,7 + 1)]
    nEdgesPerNodeTup = [10,] * len(nNodesTup)
    nReps = 10
    
    sampleComplexity = np.zeros((len(nNodesTup),))
    incorrectCounts = np.zeros((len(nNodesTup),))
    for i in range(len(nNodesTup)):
        
        nNodes = nNodesTup[i]
        nEdgesPerNode = nEdgesPerNodeTup[i]
        
        # define some params for generating a random graph
        
        # out-degree of each node in the graph
        
        alpha = .5
        
        G = nx.random_k_out_graph(nNodes,nEdgesPerNode,alpha,seed=seed)
        
        # total number of edges
        n = nNodes * nEdgesPerNode
        source = 1
        target = 0
        
        # Create the feature vectors
        featureDim = 5
        fX = np.random.multivariate_normal(np.array([1,1,7,3,2]), .5*np.eye(featureDim), size=n)
        fX = np.maximum(fX,np.zeros_like(fX))
        
        theta = np.random.multivariate_normal(np.array([2,1,1,.5,.1]), .5*np.eye(featureDim), size=1).squeeze()
        theta = np.maximum(theta,np.zeros_like(theta))
        
        weightEst = np.squeeze(fX @ theta)
        edgeToWeight = dict(zip(G.edges,[{'weight':x} for x in weightEst.tolist()]))    
        nx.set_edge_attributes(G,edgeToWeight)
        try:
            length,path = single_source_dijkstra(G,source,target,weight='weight')
        except: 
            return sampleComplexity/nReps,incorrectCounts/nReps
        
        zStar = np.zeros_like(weightEst)
        zStar[np.array(path)] = 1
        
        T = 2500
        K = 1000
          
        results = []
        incorrectCnt = 0
        for rep in range(nReps):
            
            print('Replication {} of {}'.format(rep,nReps))
            
            bandit = TransductiveBandit(fX,T,K,G,source,target,lambd_reg=0,theta=theta,v=2)
            bandit.play() 
            results.append(bandit.sampleComplexity)
            
            # Check that the result is correct
            try:
                assert(np.all(bandit.zHat==zStar))
            except:
                print('Claimed best arm: {}'.format(bandit.zHat))
                print('Best arm: {}'.format(zStar))
                print(bandit.zHat @ fX @ theta)
                print(zStar @ fX @ theta)
        
                incorrectCnt += 1
                
        sampleComplexity[i] = sum(results) 
        incorrectCounts[i] = incorrectCnt
    
    return sampleComplexity/nReps,incorrectCounts/nReps

if __name__=='__main__':
    
    scBayes,probIncorrect = runBenchmark2()
    