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
from pynfg import DecisionNode
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
    
    .. warning::
       
       This will throw an error if there is a decision node in G.starttime that 
       is not repeated throughout the net.
    
    .. note::
       
       This is the agent-approach because intelligence is assigned to a DN
       instead of being assigned to a player.
    
    """
    T0 = G.starttime
    T = G.endtime
    dnlist = [d.basename for d in G.time_partition[T0] if \
                                                isinstance(d, DecisionNode)]
    intel = {} #keys are MC iterations s, values are iq dicts
    iq = {} #keys are base names, iq timestep series 
    funcout = {} #keys are s in S, vals are eval of integrand of G(s)
    for dn in dnlist: #preallocating iq dict entries
        iq[dn] = np.zeros(T-T0+1) 
    for s in xrange(1, S+1): #sampling S sequences of policy profiles
        print s
        for t in xrange(T0, T+1): #sampling a sequence of policy profiles
            # gather list of decision nodes in time tout
            for dn in dnlist: #drawing current policy
                G.bn_part[dn][t-T0].randomCPT(mixed=mix, setCPT=True) 
                for dd in G.bn_part[dn][t-T0::]: 
                    dd.CPT = G.bn_part[dn][t-T0].CPT #apply policy to future
            for dn in dnlist: #find the iq of each player's policy in turn
                iq[dn][t-T0] = iq_calc_iter(dn, G, X, M, delta, t)
        if integrand is not None:
            funcout[s] = integrand(G) #eval integrand G(s), assign to funcout
        intel[s] = copy.deepcopy(iq)
    return intel, funcout
    
def iq_MH_iter(G, S, X, M, noise, density, delta, integrand=None, mix=False):
    """Run MH for iterSemiNFG IQ calcs
    
    :arg G: the iterated semiNFG to be evaluated
    :type G: iterSemiNFG
    :arg S: number of MH iterations
    :type S: int
    :arg noise: the degree of independence of the proposal distribution on the 
       current value.
    :type noise: float
    :arg density: the function that assigns weights to iq
    :type density: func
    :arg X: number of samples of each policy profile
    :type X: int
    :arg M: number of random alt policies to compare
    :type M: int
    :arg delta: the discount factor
    :type delta: float
    :arg integrand: a user-supplied function of G that is evaluated for each s 
       in S 
    :type integrand: func
    
    .. warning::
       
       This will throw an error if there is a decision node in G.starttime that 
       is not repeated throughout the net.

    .. note::
       
       This is the agent-approach because intelligence is assigned to a DN
       instead of being assigned to a player.
    
    """
    T0 = G.starttime
    T = G.endtime
    dnlist = [d.basename for d in G.time_partition[T0] if \
                                                isinstance(d, DecisionNode)]
    intel = {} #keys are MC iterations s, values are iq dicts
    iq = {} #keys are base names, iq timestep series
    for dn in dnlist:
        iq[dn] = np.zeros(T-T0+1) #preallocating iqs
    intel[0] = iq
    funcout = {} #keys are s in S, vals are eval of integrand of G(s)
    funcout[0] = 0
    dens = np.zeros(S+1)
    # gather list of decision nodes in base game
    for s in xrange(1, S+1): #sampling S sequences of policy profiles
        GG = copy.deepcopy(G)
        for t in xrange(T0, T+1):
            for dn in dnlist:
                GG.bn_part[dn][t-T0].CPT = G.bn_part[dn][t-T0].perturbCPT(\
                                            noise, mixed=mix, setCPT=False)
                for dd in GG.bn_part[dn][t-T0::]: 
                    dd.CPT = GG.bn_part[dn][t-T0].CPT #apply policy to future
                iq[dn][t-T0] = iq_calc_iter(dn, GG, X, M, delta, t) #getting iq
        # The MH decision
        current_dens = density(iq)
        verdict = mh_decision(current_dens, dens[s-1])
        if verdict: #accepting new CPT
            intel[s] = copy.copy(iq)
            G = copy.deepcopy(GG)
            dens[s] = current_dens
        else:
            intel[s] = intel[s-1]
            dens[s] = dens[s-1]
        if integrand is not None:
            funcout[s] = integrand(G) #eval integrand G(s), assign to funcout
        prev_dens = current_dens
    del intel[0]
    del funcout[0]
    return intel, funcout, dens
                        
def iq_calc_iter(dn, G, X, M, delta, start):
    """Calc IQ of policy at dn,start in G from the given starting point
    
    :arg dn: the basename of the DN to be evaluated
    :type dn: str
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
    T0 = G.starttime
    npvreward = 0
    p = G.bn_part[dn][start].player
    for x in xrange(1,X+1):
        G.sample() #sample from start to the end of the net
        #npv reward for the player's real policy
        npvreward = (G.npv_reward(p, start, delta) + (x-1)*npvreward)/x
    altnpv = [0]*M
    G1 = copy.deepcopy(G)
#    Ylist = [j for j in G1.partition[p] if j.time == start]
    for m in xrange(0,M): #Sample M alt policies for the player
#        for d in Ylist: 
        G1.bn_part[dn][start-T0].randomCPT(setCPT=True) #rand altpolicy for each DN in time start
        for n in G1.bn_part[dn][start-T0::]:
            n.CPT = G1.bn_part[dn][start-T0].CPT #apply altpolicy to future copies of current DN
        G1.sample() #sample altpolicy prof. to end of net
        altnpv[m] = G1.npv_reward(p, start, delta) #get alt npvreward
    worse = [j for j in altnpv if j<npvreward] #alts worse than G
    return len(worse)/M #fraction of alts worse than G is IQ                   

def mh_decision(pnew, pold, qnew=1, qold=1):
    """Decide to accept the new draw or keep the old one
    
    :arg pnew: the unnormalized likelihood of the new draw
    :type pnew: float
    :arg pold: the unnormalized likelihood of the old draw
    :type pnew: float
    :arg qnew: the probability of transitioning from the old draw to the new 
       draw.
    :type qnew: float
    :arg qold: the probability of transitioning from the new draw to the old 
       draw.
    :type qold: float
    
    """
    if pold<=0 or qnew<=0:
        a = 1
    else:
        a = min([(pnew*qold)/(pold*qnew), 1])
    u = np.random.rand()
    if a > u:
        verdict = True
    else:
        verdict = False
    return verdict