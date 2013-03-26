# -*- coding: utf-8 -*-
"""
Implements coordinated PGT intelligence for SemiNFG object

Created on Tue Mar 12 17:38:37 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""
from __future__ import division
import copy
import numpy as np
from pynfg import DecisionNode, iterSemiNFG

def iq_MC_coord(G, S, noise, X, M, innoise=1, delta=1, integrand=None, \
                mix=False, satisfice=None):
    """Run MC outer loop on random strategy sequences for SemiNFG IQ calcs
    
    :arg G: the semiNFG to be evaluated
    :type G: SemiNFG
    :arg S: number of policy profiles to sample
    :type S: int
    :type M: int
    :arg noise: the degree of independence of the proposal distribution on the 
       current value. 1 is independent, 0 returns no perturbation.
    :type noise: float
    :arg X: number of samples of each policy profile
    :type X: int
    :arg M: number of random alt policies to compare
    :arg innoise: the perturbation noise for the loop within iq_calc to draw 
       alt CPTs to compare utilities to current CPT.
    :type innoise: float
    :arg delta: the discount factor (ignored if SemiNFG)
    :type delta: float
    :arg integrand: a user-supplied function of G that is evaluated for each s 
       in S 
    :type integrand: func
    :arg pure: True if restricting sampling to pure strategies. False if mixed 
       strategies are included in sampling. Default is True.
       
    .. note::
       
       This is the coordinated-approach because intelligence is assigned to a 
       player instead of being assigned to a DecisionNode
    
    """
    intel = {} #keys are dn names, vals are iq time series
    iq = {}
    weight = {}
    w = {}
    funcout = {} #keys are s in S, vals are eval of integrand of G(s)
    for s in xrange(1, S+1): #sampling S policy profiles
        GG = copy.deepcopy(G)
        for p in GG.players:
            w[p] = 1
            for dn in GG.partition[p]: #drawing current policy
                w[p] *= dn.perturbCPT(noise, mixed=mix, returnweight=True) 
        for p in GG.players: #find the iq of each player's policy in turn
            iq[p] = iq_calc_coord(p, GG, X, M, mix, delta, innoise, satisfice)
        if integrand is not None:
            funcout[s] = integrand(GG) #eval integrand G(s), assign to funcout
        intel[s] = copy.deepcopy(iq)
        weight[s] = copy.deepcopy(w)
    return intel, funcout, weight
    
def iq_MH_coord(G, S, density, noise, X, M, innoise=1, delta=1, \
                integrand=None, mix=False, satisfice=None):
    """Run MH outer loop on random strategy sequences for SemiNFG IQ calcs
    
    :arg G: the SemiNFG to be evaluated
    :type G: SemiNFG
    :arg S: number of MH iterations
    :type S: int
    :arg density: the function that assigns weights to iq
    :type density: func
    :arg noise: the degree of independence of the proposal distribution on the 
       current value. 1 is independent, 0 returns no perturbation.
    :type noise: float
    :arg X: number of samples of each policy profile
    :type X: int
    :arg M: number of random alt policies to compare
    :type M: int
    :arg innoise: the perturbation noise for the loop within iq_calc to draw 
       alt CPTs to compare utilities to current CPT.
    :type innoise: float
    :arg delta: the discount factor (ignored if SemiNFG)
    :type delta: float
    :arg integrand: a user-supplied function of G that is evaluated for each s 
       in S 
    :type integrand: func
    :arg mix: if true, proposal distribution is over mixed CPTs. Default is 
       False.
    :type mix: bool

    .. note::
       
       This is the coordinated-approach because intelligence is assigned to a 
       player instead of being assigned to a DecisionNode
       
    """
    intel = {} #keys are s in S, vals are iq dict (dict of dicts)
    iq = {} #keys are base names, iq timestep series
    funcout = {} #keys are s in S, vals are eval of integrand of G(s)
    dens = np.zeros(S+1) #storing densities for return
    for s in xrange(1, S+1): #sampling S sequences of policy profiles
        print s
        GG = copy.deepcopy(G)
        for p in GG.players:
            for dn in GG.partition[p]: #drawing current policy 
                dn.perturbCPT(noise, mixed=mix, setCPT=False) 
        for p in GG.players:#getting iq
            iq[p] = iq_calc_coord(p, GG, X, M, mix, delta, innoise, satisfice) 
        # The MH decision
        current_dens = density(iq)
        verdict = mh_decision(current_dens, dens[s-1])
        if verdict: #accepting new CPT
            intel[s] = copy.deepcopy(iq)
            G = copy.deepcopy(GG)
            dens[s] = current_dens
        else:
            intel[s] = intel[s-1]
            dens[s] = dens[s-1]
        if integrand is not None:
            funcout[s] = integrand(G) #eval integrand G(s), assign to funcout
    return intel, funcout, dens[1::]
    
def iq_calc_coord(p, G, X, M, mix, delta, innoise, satisfice=None):
    """Calc IQ of player p in G across all of p's decision nodes
    
    :arg p: the name of the player whose intelligence is being evaluated.
    :type p: str
    :arg G: the semiNFG to be evaluated
    :type G: SemiNFG
    :arg X: number of samples of each policy profile
    :type X: int
    :arg M: number of random alt policies with which to compare
    :type M: int
    :arg delta: the discount factor (ignored if SemiNFG)
    :type delta: float
    :arg innoise: the perturbation noise for the inner loop to draw alt CPTs
    :type innoise: float
    
    """
    if isinstance(G, iterSemiNFG):
        ufoo = G.npv_reward
        uargs = [p, G.starttime, delta]
    else:
        ufoo = G.utility
        uargs = [p]
    util = 0
    for x in xrange(1,X+1):
        G.sample()
        util = (ufoo(*uargs)+(x-1)*util)/x
    altutil = [0]*M
    weight = np.ones(M)
    tick = 0
    if satisfice: #using the satisficing distribution for drawing alternatives
        G = satisfice
    for m in range(M): #Sample M alt policies for the player
        GG = copy.deepcopy(G)
        for dn in GG.partition[p]: #rand CPT for the DN
            #density for the importance sampling distribution
            if innoise == 1 or satisfice:
                dn.perturbCPT(innoise, mixed=mix)
                denw=1
            else:
                denw = dn.perturbCPT(innoise, mixed=mix, returnweight=True)
            if not tick:  
                numw = denw #scaling constant num to ~ magnitude of den
            weight[m] *= (numw/denw)
            tick += 1
        GG.sample() #sample altpolicy prof. to end of net
        if isinstance(GG, iterSemiNFG):
            altutil[m] = GG.npv_reward(p, GG.starttime, delta)
        else:
            altutil[m] = GG.utility(p)
    #weight of alts worse than G
    worse = [weight[m] for m in range(M) if altutil[m]<util]
    return np.sum(worse)/np.sum(weight) #fraction of alts worse than G is IQ
    
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