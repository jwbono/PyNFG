# -*- coding: utf-8 -*-
"""
Implements Coordinated PGT intelligence for Policies for iterSemiNFG objects

Created on Fri Mar 22 15:32:33 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""

from __future__ import division
import copy
import numpy as np
from pynfg import DecisionNode, iterSemiNFG
from pynfg.utilities.utilities import mh_decision
import sys

def policy_MC(G, S, noise, X, M, innoise=1, delta=1, integrand=None, \
                mix=False, satisfice=None):
    """Run Importance Sampling on policies for PGT Intelligence Calculations

    For examples, see below or PyNFG/bin/hideandseek.py 
    
    :arg G: the game to be evaluated
    :type G: iterSemiNFG
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
         
    .. warning::
       
       This will throw an error if there is a decision node in G.starttime that 
       is not repeated throughout the net.
       
    .. note::
       
       This is the policy-approach because intelligence is assigned to a 
       player instead of being assigned to a DecisionNode, and all DNs with 
       the same basename have the same CPT.
       
    Example::
        
        def welfare(G):
            #calculate the welfare of a single sample of the iterSemiNFG G
            G.sample()
            w = 0
            for p in G.players:
                w += G.npv_reward(p, G.starttime, 1.0)
            return w
            
        import copy
        GG = copy.deepcopy(G) #G is an iterSemiNFG
        S = 50 #number of MC samples
        X = 10 #number of samples of utility of G in calculating iq
        M = 20 #number of alternative strategies sampled in calculating iq
        noise = .2 #noise in the perturbations of G for MC sampling
        
        from pynfg.pgtsolutions.intelligence.policy import policy_MC
        
        intelMC, funcoutMC, weightMC = policy_MC(GG, S, noise, X, M, 
                                                 innoise=.2, 
                                                 delta=1, 
                                                 integrand=welfare, 
                                                 mix=False, 
                                                 satisfice=GG)
    
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
        sys.stdout.write('\r')
        sys.stdout.write('MC Sample ' + str(s))
        sys.stdout.flush()
        GG = copy.deepcopy(G)
        for p in G.players:
            w[p] = 1
            for bn in bndict[p]: #getting importance weights for each player
                w[p] *= GG.bn_part[bn][0].perturbCPT(noise, mixed=mix, \
                                                            returnweight=True)
                for dn in GG.bn_part[bn][1::]:
                    dn.CPT = GG.bn_part[bn][0].CPT
        for p in G.players: #find the iq of each player's policy in turn
            iq[p] = policy_calciq(p, GG, X, M, mix, delta, innoise, satisfice)
        if integrand is not None:
            funcout[s] = integrand(GG) #eval integrand G(s), assign to funcout
        intel[s] = copy.deepcopy(iq)
        weight[s] = copy.deepcopy(w)
    return intel, funcout, weight
    
def policy_MH(G, S, density, noise, X, M, innoise=1, delta=1, \
                integrand=None, mix=False, satisfice=None):
    """Run Metropolis-Hastings on policies for PGT Intelligence Calculations
    
    For examples, see below or PyNFG/bin/hideandseek.py
    
    :arg G: the game to be evaluated
    :type G: iterSemiNFG
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
       
    .. warning::
       
       This will throw an error if there is a decision node in G.starttime that 
       is not repeated throughout the net.

    .. note::
       
       This is the policy-approach because intelligence is assigned to a 
       player instead of being assigned to a DecisionNode, and all DNs with 
       the same basename have the same CPT.
       
    Example::
        
        def density(iqdict):
            #calculate the PGT density for a given iqdict
            x = iqdict.values()
            y = np.power(x,2)
            z = np.product(y)
            return z

        def welfare(G):
            #calculate the welfare of a single sample of the iterSemiNFG G
            G.sample()
            w = 0
            for p in G.players:
                w += G.npv_reward(p, G.starttime, 1.0)
            return w
            
        import copy
        GG = copy.deepcopy(G) #G is an iterSemiNFG
        S = 50 #number of MH samples
        X = 10 #number of samples of utility of G in calculating iq
        M = 20 #number of alternative strategies sampled in calculating iq
        noise = .2 #noise in the perturbations of G for MH sampling
        
        from pynfg.pgtsolutions.intelligence.policy import policy_MH
        
        intelMH, funcoutMH, densMH = policy_MH(GG, S, density, noise, X, M,
                                               innoise=.2, 
                                               delta=1, 
                                               integrand=welfare, 
                                               mix=False, 
                                               satisfice=GG)
       
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
        sys.stdout.write('\r')
        sys.stdout.write('MH Sample ' + str(s))
        sys.stdout.flush()
        GG = copy.deepcopy(G)
        for p in G.players: #taking the new MH draw
            for bn in bndict[p]:
                GG.bn_part[bn][0].perturbCPT(noise, mixed=mix)
                for dn in GG.bn_part[bn][1::]:
                    dn.CPT = GG.bn_part[bn][0].CPT
        for p in GG.players: #getting iq for each player with new MH draw
            iq[p] = policy_calciq(p, GG, X, M, mix, delta, innoise, satisfice) 
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
    
def policy_calciq(p, G, X, M, mix, delta, innoise, satisfice=None):
    """Estimate IQ of player's policy
    
    :arg p: the name of the player whose intelligence is being evaluated.
    :type p: str
    :arg G: the iterated semi-NFG to be evaluated
    :type G: iterSemiNFG
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
    :arg satisfice: game G such that the CPTs of G together with innoise 
       determine the intelligence satisficing distribution.
    :type satisfice: iterSemiNFG
    :returns: an estimate of the fraction of alternative policies that have a 
       lower npv reward than the current policy.
    
    """
    util = 0
    altutil = [0]*M
    weight = np.ones(M)
    tick = 0
    T0 = G.starttime
    bnlist = [x.basename for x in G.partition[p] if x.time==T0]
    for x in xrange(1,X+1):
        G.sample()
        util += G.npv_reward(p,G.starttime,delta)/X
    if satisfice: #using the satisficing distribution for drawing alternatives
        G = copy.deepcopy(satisfice)
    cptdict = G.get_decisionCPTs(mode='basename')
    smalldict = {key: cptdict[key] for key in bnlist}
    for m in range(M): #Sample M alt policies for the player
        G.set_CPTs(smalldict)
        denw = 1
        for bn in bnlist: #rand CPT for the DN
            #density for the importance sampling distribution
            if innoise==1:
                G.bn_part[bn][0].randomCPT()
            elif satisfice:
                G.bn_part[bn][0].perturbCPT(innoise, mixed=mix)
            else:
                denw *= G.bn_part[bn][0].perturbCPT(innoise, mixed=mix, \
                                                        returnweight=True)
            if not tick:  
                numw = denw #scale const. num. to ~mag. of den (save mem.)
            weight[m] *= (numw/denw)
            tick += 1
#            import pdb; pdb.set_trace()
            for dn in G.bn_part[bn][1::]:
                dn.CPT = G.bn_part[bn][0].CPT
        G.sample() #sample altpolicy prof. to end of net
        altutil[m] = G.npv_reward(p, G.starttime, delta)
    #weight of alts worse than G
    worse = [weight[m] for m in range(M) if altutil[m]<util]
    return np.sum(worse)/np.sum(weight) #fraction of alts worse than G is IQ
    
