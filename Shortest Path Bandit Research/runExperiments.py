#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 07:29:55 2020

@author: nick
"""

import numpy as np
import multiprocessing as mp
import csv

import rage
import xy_static_allocation
import oracle_static_allocation
import bayesian_iterative_allocation

def runReplicates(algList,dVals):
    
    scArray = np.zeros((len(algList),len(dVals)))
    
    for i,alg in enumerate(algList):
    
        for j,d in enumerate(dVals):
            
            Z = np.eye(d)
            xPrime = (np.cos(.01)*Z[0] + np.sin(.01)*Z[1])[None,:]
            X = np.concatenate([np.eye(d),xPrime])
            Z = X.copy()
            n = X.shape[0]
            theta = 2*Z[0]
            
            T = 1000
            K = 50
        
            initAlloc = np.ones((n,))/n
        
            # Static XY
            if i == 1:
                args = (X,Z,initAlloc,theta)
            elif i == 2:
                args = (X,Z,.05,.2,theta,initAlloc)
            elif i == 3:
                args = (X,Z,initAlloc,T,K,0,theta)
            else:
                args = (X,Z,initAlloc,theta)

            # Run the replications in parallel
            nReplications = 2
            with mp.Pool(nReplications) as _pool:
                result = _pool.map_async(runReplicate,
                                          (alg,args),
                                          callback=lambda x : print('Done!'))
                result.wait()
                resultList = result.get()
    
            # Average the sample complexity over replications
            avgSampleComplexity = float(sum(resultList))/nReplications
    
            # Store in array
            scArray[i,j] = avgSampleComplexity
            
    return scArray

def runReplicate(params):
    
    algorithm = params[0]
    args = params[1]
    
    instance = algorithm(*args)
    instance.play()
    
    return instance.sampleComplexity

if __name__ == '__main__':
    
    seed = 123456
    np.random.seed(seed)
    
    # algList = (xy_static_allocation.StaticAllocation,
    #            rage.RageBandit,
    #            bayesian_iterative_allocation.TransductiveBandit,
    #            oracle_static_allocation.OracleAllocation)
    # dVals = (5,10,15,20,25,30,35)
    algList = (rage.RageBandit,)
    dVals = (5,)
    scResults = runReplicates(algList, dVals)
    
    # Save the results to csv
    with open('benchmarkResults.csv','w') as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow(dVals)
        writer.writerows(scResults)
    
