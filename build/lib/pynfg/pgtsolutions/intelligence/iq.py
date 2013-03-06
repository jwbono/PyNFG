# -*- coding: utf-8 -*-
"""
Implements PGT intelligence for SemiNFG objects

Part of: PyNFG - a Python package for modeling and solving Network Form Games

Created on Mon Feb 25 12:07:49 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""
from __future__ import division
import copy
import numpy as np
from pynfg import DecisionNode
from pynfg.pgtsolutions.intelligence.iq_iterated import mh_decision

def iq_MC(G, S, X, M, integrand=None, mix=False):
    """Run MC outer loop on random policies for SemiNFG IQ calcs
    
    :arg G: the semiNFG to be evaluated
    :type G: SemiNFG
    :arg S: number of policy profiles to sample
    :type S: int
    :arg X: number of samples of each policy profile
    :type X: int
    :arg M: number of random alt policies to compare
    :type M: int
    :arg integrand: a user-supplied function of G that is evaluated for each s 
       in S 
    :type integrand: func
    :arg pure: True if restricting sampling to pure strategies. False if mixed 
       strategies are included in sampling. Default is True.
    
    """
    intel = {} #keys are dn names, vals are iq time series
    funcout = {} #keys are s in S, vals are eval of integrand of G(s)
    dnlist = [d.name for d in G.nodes if isinstance(d, DecisionNode)]
    for dn in dnlist:
        intel[dn] = []
    for s in xrange(S): #sampling S policy profiles
        for dn in dnlist: 
            G.node_dict[dn].randomCPT(mixed=mix, setCPT=True) #drawing current policy
        for dn in dnlist: #find the iq of each player's policy in turn
            intel[dn].append(iq_calc(dn, G, X, M))
        if integrand is not None:
            funcout[s] = integrand(G) #eval integrand G(s), assign to funcout
    return intel, funcout
    
def iq_MH(G, S, X, M, noise, dens, integrand=None, mix=False, seed=[]):
    """Run MH for SemiNFG with IQ calcs
    
    :arg G: the SemiNFG to be evaluated
    :type G: SemiNFG
    :arg S: number of MH iterations
    :type S: int
    :arg noise: the degree of independence of the proposal distribution on the 
       current value. 1 is independent, 0 returns no perturbation.
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
    :arg mix: if true, proposal distribution is over mixed CPTs. Default is 
       False.
    :type mix: bool
    :arg seed: list of names of DecisionNodes that have been seeded. DecisionNodes 
       that are not named in seed use a uniform CPT as a starting point.
    :type seed: list
       
    """
    intel = {} #keys are s in S, vals are iq dict (dict of dicts)
    dnlist = [d.name for d in G.nodes if isinstance(d, DecisionNode)]
    for dn in dnlist:
        intel[dn] = [0] #init intel dict
    funcout = {} #keys are s in S, vals are eval of integrand of G(s)
    funcout[0] = 0
    propiq = {} #dict of proposal iq, keys are names
    seedlist = [dn for dn in dnlist if dn not in seed]
    for x in seedlist: #if not seeded, DNs start uniform
            G.node_dict[dn].uniformCPT()
    for s in xrange(1,S+1): #sampling S sequences of policy profiles
        GG = copy.deepcopy(G)
        for dn in dnlist:
            GG.node_dict[dn].CPT = G.node_dict[dn].perturbCPT(noise, mixed=mix, \
                                         setCPT=False) #drawing current policy 
        for dn in dnlist: 
            propiq[dn] = iq_calc(dn, GG, X, M) #getting iq
            # The MH decision
            verdict = mh_decision(dens(propiq[dn]), dens(intel[dn][-1]))
            if verdict: #accepting new CPT
                intel[dn].append(propiq[dn])
                G.node_dict[dn].CPT = GG.node_dict[dn].CPT
            else:
                intel[dn].append(intel[dn][-1])
        if integrand is not None:
            funcout[s] = integrand(G) #eval integrand G(s), assign to funcout
    return intel, funcout

def iq_calc(dn, G, X, M):
    """Calc IQ of policy at dn in G
    
    :arg dn: the name of the DN to be evaluated
    :type dn: str
    :arg G: the semiNFG to be evaluated
    :type G: SemiNFG
    :arg X: number of samples of each policy profile
    :type X: int
    :arg M: number of random alt policies with which to compare
    :type M: int
    
    """
    p = G.node_dict[dn].player
    util = 0
    for x in xrange(1,X+1):
        G.sample()
        util = (G.utility(p)+(x-1)*util)/x
    altutil = [0]*M
    G1 = copy.deepcopy(G)
    for m in range(0,M): #Sample M alt policies for the player
        G1.node_dict[dn].randomCPT(setCPT=True) #rand for the DN
        G1.sample() #sample altpolicy prof. to end of net
        altutil[m] = G1.utility(p) #get alt utility
    worse = [j for j in altutil if j<=util] #alts worse than G
    return len(worse)/M #fraction of alts worse than G is IQ 