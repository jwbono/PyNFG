# -*- coding: utf-8 -*-
"""
Implements PGT intelligence for iterSemiNFG objects

Part of: PyNFG - a Python package for modeling and solving Network Form Games

Created on Wed Jan  2 16:33:36 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""
from __future__ import division
import copy
import numpy as np
from classes import *

def iq_MC_iter(G, S, X, M, delta, integrand=None):
    """Run MC outer loop on random policy sequences for iterSemiNFG IQ calcs
    
    :arg G: the iterated semiNFG to be evaluated
    :type G: iterSemiNFG
    :arg S: number of policy sequences to sample
    :type S: int
    :arg X: number of samples of each policy profile
    :type X: int
    :arg M: number of random alt policies to compare
    :type M: int
    :arg d: the discount factor
    :type d: float
    :arg integrand: a user-supplied function of G that is evaluated for each s 
       in S 
    :type integrand: func
    
    """
    T0 = G.starttime
    T = G.endtime
    intel = {} #keys are s in S, vals are iq dict (dict of dicts)
    iq = {} #keys are p in G.players, vals are iq time series
    funcout = {} #keys are s in S, vals are eval of integrand of G(s)
    for p in G.players: #preallocating iq dict entries
        iq[p] = [0]*(T-T0+1) 
    for s in xrange(S): #sampling S sequences of policy profiles
        for t in xrange(T0, T+1): #sampling a sequence of policy profiles
            # gather list of decision nodes in time tout
            Xlist = [d for d in G.time_partition[t] if \
                                                isinstance(d, DecisionNode)]
            for d in Xlist: 
                    d.randomCPT(setCPT=True) #drawing current policy
                    for dd in G.bn_part[d.basename][t::]: 
                        dd.CPT = d.CPT #apply policy to future timesteps
            for p in G.players: #find the iq of each player's policy in turn
                iq[p][t-T0] = iq_calc_iter(p, G, X, M, delta, t)
        if integrand is not None:
            funcout[s] = integrand(G) #eval integrand G(s), assign to funcout
        intel[s] = iq #assign iq dict to s entry in intel dict
    return intel, funcout
                        
def iq_calc_iter(p, G, X, M, delta, start):
    """Calc IQ of G from the given starting point
    
    :arg player: the player whose intelligence is to be evaluated
    :type player: str
    :arg G: the iterated semiNFG to be evaluated
    :type G: iterSemiNFG
    :arg X: number of samples of each policy profile
    :type X: int
    :arg M: number of random alt policies with which to compare
    :type M: int
    :arg d: the discount factor
    :type d: float
    :arg start: the starting time step
    :type start: int
    
    """
    npvreward = 0
    for x in range(1,X):
        G.sample_timesteps(start) #sample from start to the end of the net
        #npv reward for the player's real policy
        npvreward = (G.npv_reward(p, start, delta) + (x-1)*npvreward)/x
    altnpv = [0]*M
    G1 = copy.deepcopy(G)
    Ylist = [j for j in G1.partition[p] if j.time == start]
    for m in range(0,M): #Sample M alt policies for the player
        for d in Ylist: 
            d.randomCPT(setCPT=True) #rand altpolicy for each DN in time tout
            for n in G1.bn_part[d.basename][start::]:
                n.CPT = d.CPT #apply altpolicy to future copies of current DN
        G1.sample_timesteps(start) #sample altpolicy prof. to end of net
        altnpv[m] = G1.npv_reward(p, start, delta) #get alt npvreward
    worse = [j for j in altnpv if j<=npvreward] #alts worse than G
    return len(worse)/M #fraction of alts worse than G is IQ                   

