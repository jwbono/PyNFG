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
import scipy.stats.distributions as randvars

def iq_MC_iter(G, S, X, M, delta, integrand=None, mix=False):
    """Run MC outer loop on random policy sequences for iterSemiNFG IQ calcs
    
    :arg G: the iterated semiNFG to be evaluated
    :type G: iterSemiNFG
    :arg S: number of policy sequences to sample
    :type S: int
    :arg X: number of samples of each policy profile
    :type X: int
    :arg M: number of random alt policies to compare
    :type M: int
    :arg delta: the discount factor
    :type delta: float
    :arg integrand: a user-supplied function of G that is evaluated for each s 
       in S 
    :type integrand: func
    
    """
    T0 = G.starttime
    T = G.endtime
    intel = {} #keys are s in S, vals are iq dict (dict of dicts)
    iq = {} #keys are basenames, vals are iq time series
    funcout = {} #keys are s in S, vals are eval of integrand of G(s)
    Xlist = [d.basename for d in G.time_partition[T0] if \
                                                isinstance(d, DecisionNode)]
    for bn in Xlist: #preallocating iq dict entries
        intel[bn] = np.zeros((S,T-T0)) 
    for s in xrange(0,S): #sampling S sequences of policy profiles
        for t in xrange(T0, T+1): #sampling a sequence of policy profiles
            # gather list of decision nodes in time tout
            for bn in Xlist: 
                G.bn_part[bn][t-T0].randomCPT(mixed=mix, setCPT=True) #drawing current policy
                for dd in G.bn_part[bn][t::]: 
                    dd.CPT = G.bn_part[bn][t-T0].CPT #apply policy to future
            for bn in Xlist: #find the iq of each player's policy in turn
                intel[bn][s,t-T0] = iq_calc_iter(bn, G, X, M, delta, t)
        if integrand is not None:
            funcout[s] = integrand(G) #eval integrand G(s), assign to funcout
    return intel, funcout
    
def iq_MH_iter(G, S, X, M, noise, dens, delta, integrand=None, mix=False, \
                seed=False):
    """Run MH for iterSemiNFG IQ calcs
    
    :arg G: the iterated semiNFG to be evaluated
    :type G: iterSemiNFG
    :arg S: number of MH iterations
    :type S: int
    :arg noise: the degree of independence of the proposal distribution on the 
       current value.
    :type noise: float
    :arg dens: the function that assigns weights to iq
    :type dens: func
    :arg X: number of samples of each policy profile
    :type X: int
    :arg M: number of random alt policies to compare
    :type M: int
    :arg delta: the discount factor
    :type delta: float
    :arg integrand: a user-supplied function of G that is evaluated for each s 
       in S 
    :type integrand: func
    
    """
    T0 = G.starttime
    T = G.endtime
    intel = {} #keys are s in S, vals are iq dict (dict of dicts)
    iq = {} #keys are basenames, vals are iq time series
    funcout = {} #keys are s in S, vals are eval of integrand of G(s)
    propiq = {} #dict of proposal iq, keys are 
    # gather list of decision nodes in base game
    Xlist = [d.basename for d in G.time_partition[T0] if \
                                                isinstance(d, DecisionNode)]
    if not seed: #if not seeded, DNs start uniform
        for bn in Xlist:
            G.bn_part[bn].uniformCPT()
    for bn in Xlist: #preallocating iq dict entries
        iq[bn] = [0]*(T-T0+1) 
    for s in xrange(1,S+1): #sampling S sequences of policy profiles
        GG = copy.deepcopy(G)
        rt = randvars.randint.rvs(T0,T+1)
        rn = randvars.randint.rvs(0,len(Xlist))
        GG.bn_part[rn][rt-T0].CPT = G.bn_part[rn][rt-T0].perturbCPT(noise, \
                                                    mixed=mix, setCPT=False)
        for dd in GG.bn_part[rn][rt-T0::]: 
            dd.CPT = GG.bn_part[rn][rt-T0].CPT #apply policy to future 
        propiq[rn] = iq_calc_iter(rn, GG, X, M, delta, rt) #getting iq
        # The MH decision
        verdict = mh_decision(dens(propiq[rn]), dens(iq[s-1][rn][rt-T0]))
        if verdict: #accepting new CPT
            iq[rn][rt-T0] = propiq[rn]
            G.bn_part[rn][rt-T0].CPT = GG.bn_part[rn][rt-T0].CPT
        if integrand is not None:
            funcout[s] = integrand(G) #eval integrand G(s), assign to funcout
        intel[s] = iq #assign iq dict to s entry in intel dict
    return intel, funcout
                        
def iq_calc_iter(bn, G, X, M, delta, start):
    """Calc IQ of policy at bn,start in G from the given starting point
    
    :arg bn: the basename of the DN to be evaluated
    :type bn: str
    :arg G: the iterated semiNFG to be evaluated
    :type G: iterSemiNFG
    :arg X: number of samples of each policy profile
    :type X: int
    :arg M: number of random alt policies with which to compare
    :type M: int
    :arg delta: the discount factor
    :type delta: float
    :arg start: the starting time step
    :type start: int
    
    """
    npvreward = 0
    p = G.bn_part[bn][start].player
    for x in xrange(1,X+1):
        G.sample_timesteps(start) #sample from start to the end of the net
        #npv reward for the player's real policy
        npvreward = (G.npv_reward(p, start, delta) + (x-1)*npvreward)/x
    altnpv = [0]*M
    G1 = copy.deepcopy(G)
#    Ylist = [j for j in G1.partition[p] if j.time == start]
    for m in xrange(0,M): #Sample M alt policies for the player
#        for d in Ylist: 
        G1.bn_part[bn][start].randomCPT(setCPT=True) #rand altpolicy for each DN in time start
        for n in G1.bn_part[bn][start::]:
            n.CPT = G1.bn_part[bn][start].CPT #apply altpolicy to future copies of current DN
        G1.sample_timesteps(start) #sample altpolicy prof. to end of net
        altnpv[m] = G1.npv_reward(p, start, delta) #get alt npvreward
    worse = [j for j in altnpv if j<=npvreward] #alts worse than G
    return len(worse)/M #fraction of alts worse than G is IQ                   

def mh_decision(p,q):
    """Decide to accept the new draw or keep the old one
    
    :arg p: the unnormalized likelihood of the new draw
    :type p: float
    :arg q: the unnormalized likelihood of the old draw
    :type q: float
    
    """
    if q<=0:
        a = 1
    else:
        a = min([p/q, 1])
    u = np.random.rand()
    if a > u:
        verdict = True
    else:
        verdict = False
    return verdict