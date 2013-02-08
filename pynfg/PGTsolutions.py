# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 16:33:36 2013
Copyright (C) 2013 James Bono
GNU Affero General Public License

Part of: PyNFG - a Python package for modeling and solving Network Form Games
Implements PGT solutions for SemiNFG and iterSemiNFG objects

"""
__author__="""James Bono (jwbono@gmail.com)"""

from __future__ import division
import copy
import numpy as np
from nodes import *

def iter_iqMC(G, S, M, delta=0.75):
    intel = {}
    for p in G.players:
        intel[p] = [0]*S
        iq[p] = [0]*(G.endtime+1)
    # Sampling S sequences of policy profiles
    for s in range(S):
        # Sampling a sequence of policy profiles
        for tout in range(G.endtime+1):
            # Get a random policy for each decision node in time tout (UNIFORM, Must enable MCMC, MH, nonuniform, etc.)
            Xlist = [j for j in G.time_partition[tout] if type(j) is DecisionNode]
            for d in Xlist:
                    d.randomCPT(setCPT=True)
                    # apply the policy to each copy of that decision node
                    for n in G.basename_partition[d.basename][tout::]:
                        n.CPT = d.CPT
            # Find the intelligence of each player's policy in turn
            for p in G.players:
                iq[p][tout] = iter_iqcalc(p, G, 1, M, delta, start=tout)
        intel[p][s] = iq[p]
    return intel
                        
def iter_iqcalc(player, Game, X, M, d, start=0):
    npvreward = 0
    for x in range(X):
        # sample from tout to the end of the net using the profile
        Game.sample_timesteps(start)
        # Find the reward for the player's real policy
        npvreward = (Game.npv_reward(player, start, d) + (x-1)*npvreward)/x
    # Sample M alt policies for the player
    altnpv = [0]*M
    G1 = copy.deepcopy(G)
    Ylist = [j for j in G1.partition[player] if j.time == start]
    for m in range(0,M):
        # Get a random alt policy for each decision node in time tout
        for d in Ylist:
            d.randomCPT(setCPT=True)
            # apply the altpolicy to each copy of that decision node
            for n in G1.basename_partition[d.basename][start::]:
                n.CPT = d.CPT
        # sample from tout to the end of the net using the alt profile
        G1.sample_timesteps(start)
        altnpv[m] = G1.npv_reward(player, start, d)
    # get the intelligence of the player's real policy
    worse = [j for j in altnpv if j<=npvreward[player]]
    return len(worse)/M                    
    