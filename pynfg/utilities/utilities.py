# -*- coding: utf-8 -*-
"""
Several useful functions for use in various places in PyNFG

Part of: PyNFG - a Python package for modeling and solving Network Form Games

Created on Tue May  7 15:39:38 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""
from __future__ import division
import numpy as np
import copy

def mceu(Game, dn, N, tol=30, delta=1, verbose=False):
    """Compute the move-conditioned expected utilities for all parent values
    
    :arg Game: the SemiNFG of interest
    :type Game: SemiNFG or iterSemiNFG
    :arg dn: the name of the decision node where MCEUs are estimated
    :type dn: str
    :arg N: the max number of iterations for the estimation
    :type N: int
    :arg tol: the minimum number of samples per parent value
    :type tol: int
    
    """
    G = copy.deepcopy(Game)
    player = G.node_dict[dn].player
    CPT_shape = G.node_dict[dn].CPT.shape
    childnames = [node.name for node in G.children(dn)]
    space = G.node_dict[dn].space
    Utable = np.zeros(CPT_shape)
    visits = np.zeros(CPT_shape)
    n = 0
    try:
        ufoo = G.npv_reward
        uargs = [player, G.node_dict[dn].time, delta]
    except AttributeError:
        ufoo = G.utility
        uargs = player
    while np.min(visits)<tol and n<N:
        n += 1
        G.sample()
        idx = G.node_dict[dn].get_CPTindex()
        visits[idx[:-1]] += 1
        Utable[idx] += ufoo(*uargs)
        for a in xrange(CPT_shape[-1]):
            if a != idx[-1]:
                G.node_dict[dn].set_value(space[a])
                G.sample(start=childnames)
                idy = idx[:-1]+(a,)
                Utable[idy] += ufoo(*uargs)
    if verbose:
        print('number of unvisited messages:', \
              (visits.size-np.count_nonzero(visits))/CPT_shape[-1])
        print('least number of visits:', np.min(visits[np.nonzero(visits)])) 
    idx = (visits==0)
    visits[idx] = 1
    return Utable/visits

def convert_2_pureCPT(anarray):
    """Convert an arbitrary matrix to a pure CPT w/ weight on maximum elements
    
    :arg anarray: The numpy array to be converted to 
    :type anarray: np.array
    :returns: a normalized conditional probability distribution over actions
       given messages with all elements zero or one.
    
    """
    idx = np.where(anarray == np.max(anarray, axis=-1)[...,np.newaxis])
    newarray = np.zeros(anarray.shape)
    newarray[idx] = 1
    newCPT = newarray/np.sum(newarray, axis=-1)[...,np.newaxis]
    return newCPT
    
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