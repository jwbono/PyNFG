# -*- coding: utf-8 -*-
"""
Implements PGT intelligence for policies for SemiNFG objects

Created on Fri Mar 22 15:32:33 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""

from __future__ import division
import copy
import numpy as np
from pynfg import DecisionNode, iterSemiNFG

def iq_MC_policy(G, S, noise, X, M, innoise=1, delta=1, integrand=None, \
                mix=False, satisfice=None):
    """Run MC outer loop on random policies for SemiNFG IQ calcs
    
    :arg G: the semiNFG to be evaluated
    :type G: SemiNFG
    :arg S: number of policy profiles to sample
    :type S: int
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
    :arg mix: False if restricting sampling to pure strategies. True if mixed 
       strategies are included in sampling. Default is False.
    :type mix: bool
    :arg satisfice: game G such that the CPTs of G together with innoise 
       determine the intelligence satisficing distribution.
    :type satisfice: iterSemiNFG
    :returns: 
       * intel - a sample-keyed dictionary of player-keyed iq dictionaries
       * funcout - a sample-keyed dictionary of the output of the 
          user-supplied integrand.
       * weight - a sample-keyed dictionay of player-keyed importance weight
          dictionaries.
       
    .. note::
       
       This is the coordinated-approach because intelligence is assigned to a 
       player instead of being assigned to a DecisionNode
    
    """
    intel = {} #keys are sample indices, vals are iq dictionaries
    iq = {} #keys are player names, vals are iqs
    weight = {} #keys are 
    w = {}
    funcout = {} #keys are s in S, vals are eval of integrand of G(s)
    bndict = {}
    T0 = G.starttime
    for p in G.players: #getting player-keyed dict of basenames
        bndict[p] = [x.basename for x in G.partition[p] if x.time==T0]
    for s in xrange(1, S+1): #sampling S policy profiles
        print s
        GG = copy.deepcopy(G)
        for p in G.players:
            w[p] = 1
            for bn in bndict[p]: #getting importance weights for each player
                w[p] *= GG.bn_part[bn][T0].perturbCPT(noise, mixed=mix, \
                                                            returnweight=True)
#                for dn in GG.bn_part[bn][T0+1:]:
#                    dn.CPT = GG.bn_part[bn][T0].CPT
        for p in G.players: #find the iq of each player's policy in turn
            iq[p] = iq_calc_policy(p, GG, X, M, mix, delta, innoise, satisfice)
        if integrand is not None:
            funcout[s] = integrand(GG) #eval integrand G(s), assign to funcout
        intel[s] = copy.deepcopy(iq)
        weight[s] = copy.deepcopy(iq)
    return intel, funcout, weight
    
def iq_MH_policy(G, S, density, noise, X, M, innoise=1, delta=1, \
                integrand=None, mix=False, satisfice=None):
    """Run MH for SemiNFG with IQ calcs
    
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
    :arg satisfice: game G such that the CPTs of G together with innoise 
       determine the intelligence satisficing distribution.
    :type satisfice: iterSemiNFG
    :returns: 
       * intel - a sample-keyed dictionary of player-keyed iq dictionaries
       * funcout - a sample-keyed dictionary of the output of the 
          user-supplied integrand.
       * dens - a list of the density values, one for each MH draw.

    .. note::
       
       This is the coordinated-approach because intelligence is assigned to a 
       player instead of being assigned to a DecisionNode
       
    """
    intel = {} #keys are s in S, vals are iq dict (dict of dicts)
    iq = {} #keys are base names, iq timestep series
    funcout = {} #keys are s in S, vals are eval of integrand of G(s)
    dens = np.zeros(S+1) #storing densities for return
    bndict = {} #mapping from player name to DN basenames
    T0 = G.starttime
    for p in G.players: #getting player-keyed dict of basenames
        bndict[p] = [x.basename for x in G.partition[p] if x.time==T0]
    for s in xrange(1, S+1): #sampling S sequences of policy profiles
        print s
        GG = copy.deepcopy(G)
        for p in G.players: #taking the new MH draw
            for bn in bndict[p]:
                GG.bn_part[bn][T0].perturbCPT(noise, mixed=mix) 
        for p in GG.players: #getting iq for each player with new MH draw
            iq[p] = iq_calc_policy(p, GG, X, M, mix, delta, innoise, satisfice) 
        # The MH decision
        current_dens = density(iq) #evaluating density of current draw's iq
        verdict = mh_decision(current_dens, dens[s-1]) #True if accept new draw
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
    
def iq_calc_policy(p, G, X, M, mix, delta, innoise, satisfice=None):
    """Calc IQ of player p in G across all of p's decision nodes
    
    :arg p: the name of the player whose intelligence is being evaluated.
    :type p: str
    :arg G: the semiNFG to be evaluated
    :type G: SemiNFG
    :arg X: number of samples of each policy profile
    :type X: int
    :arg M: number of random alt policies with which to compare
    :type M: int
    :arg mix: if true, proposal distribution is over mixed CPTs. Default is 
       False.
    :type mix: bool
    :arg delta: the discount factor (ignored if SemiNFG)
    :type delta: float
    :arg innoise: the perturbation noise for the inner loop to draw alt CPTs
    :type innoise: float
    :returns: the fraction of alternative policies for the given player that 
       have a lower npv reward than the current policy.
    
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
    T0 = G.starttime
    bnlist = [x.basename for x in G.partition[p] if x.time==T0]
    if satisfice: #using the satisficing distribution for drawing alternatives
        G = satisfice
    for m in range(M): #Sample M alt policies for the player
        GG = copy.deepcopy(G)
        denw = 1
        for bn in bnlist: #rand CPT for the DN
            #density for the importance sampling distribution
            if innoise == 1 or satisfice:
                GG.bn_part[bn][T0].perturbCPT(innoise, mixed=mix)
            else:
                denw *= GG.bn_part[bn][T0].perturbCPT(innoise, mixed=mix, \
                                                        returnweight=True)
            if not tick:  
                numw = denw #scaling constant num to ~ magnitude of den
            weight[m] *= (numw/denw)
            tick += 1
#            import pdb; pdb.set_trace()
#            for dn in GG.bn_part[bn][T0+1:]:
#                dn.CPT = GG.bn_part[bn][T0].CPT
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
    :returns: either True or False to determine whether the new draw is 
       accepted.
    
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