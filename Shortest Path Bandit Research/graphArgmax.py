#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 11:30:42 2021

@author: nick
"""

import numpy as np
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra

class ShortestPathFinder:
    
    def __init__(self,graph,source,target):
        
        self.graph = graph
        
        # The source and target of our shortest path finding
        self.source = source
        self.target = target

    def getArgmax(self,theta):
          
        theta = theta.T
        
        if len(theta.shape) > 1:
            
            lengths,paths = np.zeros(theta.shape[0]),np.zeros_like(theta)
            
            for i in range(theta.shape[0]):
                length,path = self._getArgmax(theta[i])
                lengths[i],paths[i,np.array(path)] = length,1
        
            length,path = np.stack(lengths),np.stack(paths)
        
        else:
            length,_path = self._getArgmax(theta)
            path = np.zeros_like(theta)
            path[np.array(_path)] = 1
    
        return length,path.squeeze().T

    def _getArgmax(self,theta):
        
        _theta = theta.copy()
        if np.any(theta<0):
            _theta += -np.min(theta)
        
        edgeToWeight = dict(zip(self.graph.edges,
                                [{'weight':x} for x in _theta.tolist()]))
        
        nx.set_edge_attributes(self.graph,edgeToWeight)
        
        try:    
            length,path = single_source_dijkstra(self.graph,self.source,self.target,weight='weight')
        except Exception as e:
            print(e)
            raise e
    
        return length,path

if __name__ == "__main__":

    seed = 13648  # Seed random number generators for reproducibility
    
    G = nx.random_k_out_graph(10, 3, 0.5, seed=seed)
    
    theta = np.random.normal(10,1,len(G.edges))
    edgeToWeight = dict(zip(G.edges,[{'weight':x} for x in theta.tolist()]))
    nx.set_edge_attributes(G, edgeToWeight)
