# -*- coding: utf-8 -*-
"""
Implements Uncoordinated PGT Intelligence for SemiNFG objects

Created on Wed Jan  2 16:33:36 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""
from __future__ import division
import copy
import numpy as np
from pynfg import DecisionNode
from pynfg import iterSemiNFG
import scipy.stats.distributions as randvars
from pynfg.utilities.utilities import mh_decision
import sys

def uncoordinated_MC(G, S, noise, X, M, innoise, delta=1, integrand=None, \
                     mix=False, satisfice=None):
    """Run Importance Sampling on strategies for PGT Intelligence Calculations
    
    For examples, see below or PyNFG/bin/stackelberg.py for SemiNFG or 
    PyNFG/bin/hideandseek.py for iterSemiNFG
    
    :arg G: the game to be evaluated
    :type G:  SemiNFG or iterSemiNFG
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
    :type satisfice: SemiNFG or iterSemiNFG
    :returns: 
       * intel - a sample-keyed dictionary of decision node-keyed iq dicts
       * funcout - a sample-keyed dictionary of the output of the 
         user-supplied integrand.
       * weight - a sample-keyed dictionay of decision nod-keyed importance 
         weight dictionaries.
    
    .. note::
       
       This is the uncoordinated approach because intelligence is assigned to 
       decision nodes instead of being assigned to players. As a result, it 
       takes much longer to run than 
       pynfg.pgtsolutions.intelligence.coordinated.coordinated_MC
       
    Example::
        
        def welfare(G):
            #calculate the welfare of a single sample of the SemiNFG G
            G.sample()
            w = G.utility('1')+G.utility('2') #'1' & '2' are player names in G
            return w
            
        import copy
        GG = copy.deepcopy(G) #G is a SemiNFG
        S = 50 #number of MC samples
        X = 10 #number of samples of utility of G in calculating iq
        M = 20 #number of alternative strategies sampled in calculating iq
        noise = .2 #noise in the perturbations of G for MC sampling
        
        from pynfg.pgtsolutions.intelligence.uncoordinated import uncoordinated_MC
        
        intelMC, funcoutMC, weightMC = uncoordinated_MC(GG, S, noise, X, M, 
                                                        innoise=.2, 
                                                        delta=1, 
                                                        integrand=welfare, 
                                                        mix=False, 
                                                        satisfice=GG)
    
    """
    dnlist = [d.name for d in G.nodes if isinstance(d, DecisionNode)]
    intel = {} #keys are MC iterations s, values are iq dicts
    iq = dict(zip(dnlist, np.zeros(len(dnlist)))) #keys are node names, vals are iqs
    funcout = {} #keys are s in S, vals are eval of integrand of G(s)
    w = {}
    weight = {}
    for s in xrange(1, S+1): #sampling S sequences of policy profiles
        sys.stdout.write('\r')
        sys.stdout.write('MC Sample ' + str(s))
        sys.stdout.flush()
        GG = copy.deepcopy(G)
        for dn in dnlist: #drawing current policy
            w[dn] = GG.node_dict[dn].perturbCPT(noise, mixed=mix, \
                                                returnweight=True)
        for dn in dnlist: #find the iq of each player's policy in turn
            iq[dn] = uncoordinated_calciq(dn, GG, X, M, mix, delta, innoise, \
                                          satisfice)
        if integrand is not None:
            funcout[s] = integrand(GG) #eval integrand GG(s), assign to funcout
        intel[s] = copy.deepcopy(iq)
        weight[s] = copy.deepcopy(w)
    return intel, funcout, weight
    
def uncoordinated_MH(G, S, density, noise, X, M, innoise=1, delta=1, \
                     integrand=None, mix=False, satisfice=None):
    """Run Metropolis-Hastings on strategies for PGT Intelligence Calculations
    
    For examples, see below or PyNFG/bin/stackelberg.py for SemiNFG or 
    PyNFG/bin/hideandseek.py for iterSemiNFG
    
    :arg G: the game to be evaluated
    :type G: SemiNFG or iterSemiNFG
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
    :type satisfice: SemiNFG or iterSemiNFG
    :returns: 
       * intel - a sample-keyed dictionary of decision node-keyed iq dicts
       * funcout - a sample-keyed dictionary of the output of the
         user-supplied integrand.
       * dens - a list of the density values, one for each MH draw.

    .. note::
       
       This is the uncoordinated approach because intelligence is assigned to 
       decision nodes instead of being assigned to players. As a result, it 
       takes much longer to run than 
       pynfg.pgtsolutions.intelligence.coordinated.coordinated_MH
       
    Example::
        
        def density(iqdict):
            #calculate the PGT density for a given iqdict
            x = iqdict.values()
            y = np.power(x,2)
            z = np.product(y)
            return z

        def welfare(G):
            #calculate the welfare of a single sample of the SemiNFG G
            G.sample()
            w = G.utility('1')+G.utility('2') #'1' & '2' are player names in G
            return w
            
        import copy
        GG = copy.deepcopy(G) #G is a SemiNFG
        S = 50 #number of MH samples
        X = 10 #number of samples of utility of G in calculating iq
        M = 20 #number of alternative strategies sampled in calculating iq
        noise = .2 #noise in the perturbations of G for MH sampling
        
        from pynfg.pgtsolutions.intelligence.uncoordinated import uncoordinated_MH
        
        intelMH, funcoutMH, densMH = uncoordinated_MH(GG, S, density, noise, 
                                                      X, M,
                                                      innoise=.2, 
                                                      delta=1, 
                                                      integrand=welfare, 
                                                      mix=False, 
                                                      satisfice=GG)
    
    """
    dnlist = [d.name for d in G.nodes if isinstance(d, DecisionNode)]
    intel = {} #keys are s in S, vals are iq dict (dict of dicts)
    iq = {} #keys are base names, iq timestep series
    funcout = {} #keys are s in S, vals are eval of integrand of G(s)
    dens = np.zeros(S+1) #storing densities for return
    for s in xrange(1, S+1): #sampling S sequences of policy profiles
        sys.stdout.write('\r')
        sys.stdout.write('MH Sample ' + str(s))
        sys.stdout.flush()
        GG = copy.deepcopy(G)
        for dn in dnlist:
            GG.node_dict[dn].perturbCPT(noise, mixed=mix) 
        for dn in dnlist:#getting iq
            iq[dn] = uncoordinated_calciq(dn, GG, X, M, mix, delta, innoise, \
                                          satisfice) 
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
            funcout[s] = integrand(GG) #eval integrand G(s), assign to funcout
    return intel, funcout, dens[1::]
                        
def uncoordinated_calciq(dn, G, X, M, mix, delta, innoise, satisfice=None):
    """Estimate IQ of policy at the current decision node
    
    :arg p: the name of the player whose intelligence is being evaluated.
    :type p: str
    :arg G: the iterated semi-NFG to be evaluated
    :type G: SemiNFG or iterSemiNFG
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
    :type satisfice: SemiNFG or iterSemiNFG
    :returns: an estimate of the fraction of alternative strategies that yield 
       lower expected utility than the current policy.
    
    """
    util = 0
    altutil = [0]*M
    weight = np.ones(M)
    tick = 0
    p = G.node_dict[dn].player
    oldCPT = copy.copy(G.node_dict[dn].CPT)
    try:
        ufoo = G.npv_reward
        uargs = [p, G.starttime, delta]
    except AttributeError:
        ufoo = G.utility
        uargs = p
    for x in xrange(1,X+1):
        G.sample()
        util = (ufoo(*uargs)+(x-1)*util)/x
    if satisfice: #using the satisficing distribution for drawing alternatives
        G = copy.deepcopy(satisfice)
    oldcpt = G.bn_part[dn].CPT
    for m in range(M): #Sample M alt CPTs for the player at the DN
        G.bn_part[dn].CPT = oldcpt
        if innoise == 1 or satisfice:
            G.node_dict[dn].perturbCPT(innoise, mixed=mix)
            denw=1
        else:
            denw = G.node_dict[dn].perturbCPT(innoise, mixed=mix, \
                                               returnweight=True)
        if not tick:  
            numw = denw #scaling constant num to ~ magnitude of den
        weight[m] *= (numw/denw)
        tick += 1
        G.sample() #sample altpolicy prof. to end of net
        try:
            altutil[m] = G.npv_reward(p, GG.starttime, delta)
        except AttributeError:
            altutil[m] = G.utility(p)
        G.node_dict[dn].CPT = oldCPT #resetting the CPT for the next draw
    #weight of alts worse than G
    worse = [weight[m] for m in range(M) if altutil[m]<util]
    return np.sum(worse)/np.sum(weight) #fraction of alts worse than G is IQ