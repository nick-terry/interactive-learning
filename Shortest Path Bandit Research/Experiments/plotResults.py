#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 11:06:20 2021

@author: nick
"""

import matplotlib.pyplot as plt
import csv

with open('benchmarkResults.csv','r') as f:
    reader = csv.reader(f)\
    
    i = 1
    data = []
    for row in reader:
        if i == 1:
            dVals = tuple(map(int,row))
        else:
            data.append(tuple(map(float,row)))
        
        i += 1

labels = ('Static XY-Allocation','RAGE','Bayesian alg.','Oracle Static Allocation')

fig,ax = plt.subplots(1)
for row,label in zip(data,labels):
    ax.plot(dVals,row,label=label)
ax.legend()
ax.set_xlabel('d')
ax.set_ylabel('Sample Complexity')
ax.set_yscale('log')
lowerLim = 5
ax.set_ylim([10**lowerLim,10**8])
ax.set_yticks([10**k for k in range(lowerLim,9)])

plt.savefig('benchmark.png',dpi=150)