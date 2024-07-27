#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:17:58 2023

@author: ericbell
"""


import itertools
import networkx
from sklearn.linear_model import LinearRegression as LR
import numpy as np
import csv
import pandas as pd
from scipy.stats import norm

def nump(arr):
    return np.transpose(np.array(arr))

        
def comp_partcorr(df,i,j,S):
    indi, indj = i-1, j-1
    xi, xj = df.loc[:,indi], df.loc[:,indj]
    ri, rj = xi.copy(),xj.copy()
    
    if not S: return np.corrcoef(ri,rj)
    
    
    xs = df.iloc[:,S]
    ireg = LR().fit(xs,xi)
    jreg = LR().fit(xs,xj)
    
    ri = xi - ireg.predict(xs)
    rj = xj - jreg.predict(xs)
    
    return np.corrcoef(ri,rj)


def fisher(n,p):
    
    return np.sqrt(n-2-3)*np.arctanh(p)


def pval(z):
    return 2 * norm.cdf(-np.abs(z))

import networkx as nx

def construct_graph(π, alpha):
    # Compute partial correlations for each pair of variables
    partial_correlations = np.zeros((len(π), len(π)))
    for i in range(len(π)):
        for j in range(i + 1, len(π)):
            corr = np.corrcoef(π[i], π[j])[0, 1]
            t_value = corr / np.sqrt((1 - corr**2) / (len(π) - 2))
            p_value = 2 * (1 - norm.cdf(np.abs(t_value)))
            if p_value > alpha:
                partial_correlations[i, j] = corr
                partial_correlations[j, i] = corr
    
    # Create NetworkX graph from partial correlations
    Gπ = nx.Graph()
    for i in range(len(π)):
        for j in range(i + 1, len(π)):
            if partial_correlations[i, j] != 0:
                Gπ.add_edge(i, j)
    
    return Gπ
