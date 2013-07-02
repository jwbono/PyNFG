# -*- coding: utf-8 -*-
"""
Implements Uncoordinated PGT Intelligence for iterSemiNFG objects

Created on Wed Jan  2 16:33:36 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""
from __future__ import division
import copy
import numpy as np
from pynfg import DecisionNode
from pynfg.utilities.utilities import mh_decision
import scipy.stats.distributions as randvars
import sys

def iterated_MC(G, S, noise, X, M, innoise=1, delta=1, integrand=None, \
                mix=False, satisfice=None):
    """Run Importance Sampling on policy sequences for PGT IQ Calculations
    
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
       * intel - a sample-keyed dictionary of basename-keyed timestep iq lists
       * funcout - a sample-keyed dictionary of the output of the 
         user-supplied integrand.
       * weight - a sample-keyed dictionay of basename-keyed importance weight
         dictionaries.
    
    .. warning::
       
       This will throw an error if there is a decision node in G.starttime that 
       is not repeated throughout the net.
    
    .. note::
       
       This is an uncoordinated approach because intelligence is assigned to a 
       decision node instead of players. As a result, it takes much longer to 
       run than pynfg.pgtsolutions.intelligence.policy.policy_MC
       
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
        
        from pynfg.pgtsolutions.intelligence.iterated import iterated_MC
        
        intelMC, funcoutMC, weightMC = iterated_MC(GG, S, noise, X, M, 
                                                   innoise=.2, 
                                                   delta=1, 
                                                   integrand=welfare, 
                                                   mix=False, 
                                                   satisfice=GG)
    
    """
    T0 = G.starttime
    T = G.endtime
    bnlist = [d.basename for d in G.time_partition[T0] if \
                                                isinstance(d, DecisionNode)]
    intel = {} #keys are MC iterations s, values are iq dicts
    iq = {} #keys are base names, iq timestep series 
    funcout = {} #keys are s in S, vals are eval of integrand of G(s)
    weight = {} #keys are s in S, vals are bn-keyed dicts of importance weights
    for bn in bnlist: #preallocating iq dict entries
        iq[bn] = np.zeros(T-T0+1) 
    for s in xrange(1, S+1): #sampling S sequences of policy profiles
        sys.stdout.write('\r')
        sys.stdout.write('MC Sample ' + str(s))
        sys.stdout.flush()
        GG = copy.deepcopy(G)
        w = dict(zip(bnlist, np.ones(len(bnlist)))) #mapping bn to IS weights
        for t in xrange(T0, T+1): #sampling a sequence of policy profiles
            # gather list of decision nodes in time tout
            for bn in bnlist: #drawing current policy
                w[bn] *= GG.bn_part[bn][t-T0].perturbCPT(noise, mixed=mix, \
                                                            returnweight=True) 
                for dd in GG.bn_part[bn][t-T0+1::]: 
                    dd.CPT = GG.bn_part[bn][t-T0].CPT #apply policy to future
            for bn in bnlist: #find the iq of each player's policy in turn
                iq[bn][t-T0] = iterated_calciq(bn, GG, X, M, mix, delta, t, \
                                               innoise, satisfice=None) #getting iq
        if integrand is not None:
            funcout[s] = integrand(GG) #eval integrand G(s), assign to funcout
        intel[s] = copy.deepcopy(iq)
        weight[s] = copy.deepcopy(w)
    return intel, funcout, weight
    
def iterated_MH(G, S, density, noise, X, M, innoise=1, delta=1, \
                integrand=None, mix=False, satisfice=None):
    """Run Metropolis-Hastings on policy sequences for PGT IQ Calculations
    
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
       * intel - a sample-keyed dictionary of basename-keyed timestep iq lists
       * funcout - a sample-keyed dictionary of the output of the
         user-supplied integrand.
       * dens - a list of the density values, one for each MH draw.
    
    .. warning::
       
       This will throw an error if there is a decision node in G.starttime that 
       is not repeated throughout the net.

    .. note::
       
       This is an uncoordinated approach because intelligence is assigned to a 
       decision node instead of players. As a result, it takes much longer to 
       run than pynfg.pgtsolutions.intelligence.policy.policy_MH
       
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
        
        from pynfg.pgtsolutions.intelligence.iterated import iterated_MH
        
        intelMH, funcoutMH, densMH = iterated_MH(GG, S, density, noise, X, M,
                                                 innoise=.2, 
                                                 delta=1, 
                                                 integrand=welfare, 
                                                 mix=False, 
                                                 satisfice=GG)
    
    """
    T0 = G.starttime
    T = G.endtime
    dnlist = [d.basename for d in G.time_partition[T0] if \
                                                isinstance(d, DecisionNode)]
    intel = {} #keys are MC iterations s, values are iq dicts
    iq = {} #keys are base names, iq timestep series
    for dn in dnlist:
        iq[dn] = np.zeros(T-T0+1) #preallocating iqs
    funcout = {} #keys are s in S, vals are eval of integrand of G(s)
    dens = np.zeros(S+1)
    # gather list of decision nodes in base game
    for s in xrange(1, S+1): #sampling S sequences of policy profiles
        sys.stdout.write('\r')
        sys.stdout.write('MH Sample ' + str(s))
        sys.stdout.flush()
        GG = copy.deepcopy(G)
        for t in xrange(T0, T+1):
            for dn in dnlist:
                GG.bn_part[dn][t-T0].CPT = G.bn_part[dn][t-T0].perturbCPT(\
                                            noise, mixed=mix)
                for dd in GG.bn_part[dn][t-T0+1::]: 
                    dd.CPT = GG.bn_part[dn][t-T0].CPT #apply policy to future
                iq[dn][t-T0] = iterated_calciq(dn, G, X, M, mix, delta, t, \
                                               innoise, satisfice=None) #getting iq
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
                        
def iterated_calciq(bn, G, X, M, mix, delta, start, innoise, satisfice=None):
    """Estimate IQ of player's policy at a given time step
    
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
    :returns: an estimate of the fraction of alternative policies at the given 
       time step that have a lower npv reward than the current policy.
    
    """
    T0 = G.starttime
    p = G.bn_part[bn][start-T0].player
    util = 0
    altutil = [0]*M
    weight = np.ones(M)
    tick = 0
    bnlist = [x.name for x in G.bn_part[bn]]
    for x in xrange(1,X+1):
        G.sample()
        util += G.npv_reward(p,start,delta)/X
    if satisfice: #using the satisficing distribution for drawing alternatives
        G = copy.deepcopy(satisfice)
    cptdict = G.get_decisionCPTs()
    smalldict = {name: cptdict[name] for name in bnlist}
    for m in range(M): #Sample M alt policies for the player
        G.set_CPTs(smalldict)
        denw = 1
        #density for the importance sampling distribution
        if innoise == 1 or satisfice:
            G.bn_part[bn][start-T0].perturbCPT(innoise, mixed=mix)
        else:
            denw = G.bn_part[bn][start-T0].perturbCPT(innoise, mixed=mix, \
                                                        returnweight=True)
        if not tick:  
            numw = denw #scaling constant num to ~ magnitude of den
        weight[m] = (numw/denw)
        tick += 1
#       import pdb; pdb.set_trace()
        for dn in G.bn_part[bn][start-T0+1::]:
            dn.CPT = G.bn_part[bn][start-T0].CPT
        G.sample_timesteps(T0) #sample altpolicy prof. to end of net
        altutil[m] = G.npv_reward(p, start, delta)
    #weight of alts worse than G
    worse = [weight[m] for m in range(M) if altutil[m]<util]
    return np.sum(worse)/np.sum(weight) #fraction of alts worse than G is IQ