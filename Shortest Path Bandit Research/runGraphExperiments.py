#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 09:48:41 2021

@author: nick
"""
import numpy as np
import multiprocessing as mp
import csv
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra

import bayesian_allocation_graph as bag

seed = 123456

def generateGraph(nNodes,nEdgesPerNode,alpha,weights,source,target):
    
    # make graph
    G = nx.random_k_out_graph(nNodes,nEdgesPerNode,alpha)
    
    # verify that it has a path from source to target
    edgeToWeight = dict(zip(G.edges,[{'weight':x} for x in weights.tolist()]))    
    nx.set_edge_attributes(G,edgeToWeight)
    try:
        length,path = single_source_dijkstra(G,source,target,weight='weight')
        zStar = np.zeros_like(weights)
        zStar[np.array(path)] = 1
    except: 
        G,zStar = None,None
    
    return G,zStar

def runReplicates(nList,kList):
    
    scArray = np.zeros((len(nList),))
    
    for i,gTup in enumerate(zip(nList,kList)):
        
        nNodes,nEdgesPerNode = gTup
        alpha = .5
        
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
        
        weights = fX @ theta
        G,zStar = generateGraph(nNodes, nEdgesPerNode, alpha, weights, source, target)
        
        # If the graph doesnt have the path we want, retry
        j = 0
        while G is None:
            G,zStar = generateGraph(nNodes, nEdgesPerNode, alpha, weights, source, target)
            j += 1
            if j > 10:
                raise Exception
        
        T = 2500
        K = 1000

        params = (fX,T,K,G,source,target,0,-theta,2)

        # Run the replications in parallel
        nReplications = 10
        with mp.Pool(nReplications) as _pool:
            result = _pool.map_async(runReplicate,
                                      [params,]*nReplications,
                                      callback=lambda x : print('Done!'))
            result.wait()
            resultList = result.get()

        # Average the sample complexity over replications
        avgSampleComplexity = float(sum(resultList))/nReplications

        # Store in array
        scArray[i] = avgSampleComplexity
            
    return scArray

def runReplicate(params):
    
    algorithm = bag.TransductiveBandit
    
    instance = algorithm(*params)
    instance.play()
    
    return instance.sampleComplexity

if __name__ == '__main__':

    # Run several different size graphs w/ feature vectors for each edge    
    np.random.seed(seed)

    nNodesTup = [2**i for i in range(3,7 + 1)]
    nEdgesPerNodeTup = [10,] * len(nNodesTup)
    
    scResults = runReplicates(nNodesTup,nEdgesPerNodeTup)
    
    # Save the results to csv
    with open('graphBenchmarkResults.csv','w') as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow(nNodesTup)
        writer.writerow(nEdgesPerNodeTup)
        writer.writerows(scResults)