# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 19:33:44 2013

@author: jamesbono
"""

from __future__ import division
import copy
import numpy as np
import matplotlib.pylab as plt
from nodes import get_CPTindex, dict2list_vals

def ewma_jaakkola(G, bn, J, N, alpha, delta, eps):
    # getting shorter/more descriptive variable names to work with
    T0 = G.starttime
    T = G.endtime+1
    player = G.basename_partition[bn][T0].player
    shape = G.basename_partition[bn][T0].CPT.shape
    shape_last = shape[-1]
    # initializing RL parameters
    visit = set()
    R = 0
    A = 0
    B = {}
    D = {}
    Q = np.zeros(shape)
    V = np.zeros(shape[:-1])
    C = {}
    I = {}
    Rseries = np.zeros(N)
    for n in xrange(N):
        print n
        # visitn must be cleared at the start of every episode
        indicaten = np.zeros(Q.shape)
        visitn = set()
        Rseries[n] = R
        A *= alpha
        for j in xrange(J):
            # visitj must be cleared at the start of every run
            visitj = set()
            for t in xrange(T0,T):
                G.basename_partition[bn][t].CPT = G.basename_partition[bn][T0].CPT
                G.sample_timesteps(t, t)
                rew = G.reward(player, t)
#                parnodes = nodelist[t].parents.values()
                malist = dict2list_vals(G.basename_partition[bn][t].parents, \
                                            valueinput=G.basename_partition[bn][t].value)
                mapair = get_CPTindex(G.basename_partition[bn][t], malist)
                # updating scalar dynamics
                a = A
                A = 1+a
                r = R
                R = (1/A)*(a*r+rew)
                # updating set of visited (m,a) pairs
                xm = set()
                for values in visitj:
                    # past values
                    b = B[values]
                    d = D[values]
                    q = Q[values]
                    c = C[values]
                    # update equations double letters are time t
                    bb = (b+1)
                    dd = d+1
                    qq = (1/dd)*(d*q+(delta**(bb-1))*(rew-r))    
                    cc = (d/dd)*c+(1/dd)*(delta**(bb-1))
                    qq += -cc*(R-r)
                    # update dictionaries
                    B[values] = bb
                    D[values] = dd
                    Q[values] = qq
                    C[values] = cc
                    # value function
                    message = values[:-1]
                    if message not in xm:
                        # past values
                        b = B[message]
                        d = D[message]
                        v = V[message]
                        c = C[message]
                        # current
                        xj = True
                        xn = True
                        # update equations double letters are time t
                        bb = (b+1)
                        dd = d+1
                        vv = (1/dd)*(d*v+(delta**(bb-1))*(rew-r))    
                        cc = (d/dd)*c+(1/dd)*(delta**(bb-1))
                        vv += -cc*(R-r)
                        # update dictionaries
                        B[message] = bb
                        D[message] = dd
                        V[message] = vv
                        C[message] = cc
                        xm.add(message)
                if mapair not in visitj:
                    message = mapair[:-1]
                    messtrue = (message not in xm)
                    B[mapair] = 1
                    if mapair not in visitn and mapair not in visit:
                        D[mapair] = 1
                        Q[mapair] = rew-r
                        C[mapair] = 1
                        if messtrue:
                            D[message] = 1
                            V[message] = rew-r
                            C[message] = 1
                    elif mapair not in visitn:
                        D[mapair] = alpha*D[mapair]+1
                        Q[mapair] = (1/D[mapair])*((D[mapair]-1)*Q[mapair]\
                                    +(rew-r))
                        C[mapair] = 1/D[mapair]
                        if messtrue:
                            D[message] = alpha*D[message]+1
                            V[message] = (1/D[message])*((D[message]-1)*\
                                        V[message]+(rew-r))
                            C[message] = 1/D[message]
                    else:
                        D[mapair] += 1
                        Q[mapair] = (1/D[mapair])*((D[mapair]-1)*Q[mapair]\
                                    +(rew-r))
                        C[mapair] = ((D[mapair]-1)/D[mapair])*C[mapair]\
                                    +(1/D[mapair])
                        if messtrue:
                            D[message] += 1
                            V[message] = (1/D[message])*((D[message]-1)*\
                                        V[message]+(rew-r))
                            C[message] = ((D[message]-1)/D[message])*C[message]\
                                            +(1/D[message])
                    Q[mapair] += -C[mapair]*(R-r)
                    if messtrue:
                        B[message] = 1
                        V[message] += -C[message]*(R-r)
                    # mapair gets added to visit sets the first time it appears
                    visit.add(mapair)
                    visitn.add(mapair)
                    indicaten[message] = 1
                    visitj.add(mapair)
        # update CPT with shift towards Qtable argmax actions.
        shift = Q-V[...,np.newaxis]
        idx = np.nonzero(shift)
        shiftnorm = np.absolute(shift).max(axis=-1)[...,np.newaxis]
        updater = eps*indicaten*G.basename_partition[bn][T0].CPT/shiftnorm
        G.basename_partition[bn][T0].CPT[idx] += updater[idx]*(Q-V[...,np.newaxis])[idx]
        CPTsum = G.basename_partition[bn][T0].CPT.sum(axis=-1)
        G.basename_partition[bn][T0].CPT = G.basename_partition[bn][T0].CPT/CPTsum[...,np.newaxis]
        if np.any(G.basename_partition[bn][T0].CPT<0):
            raise AssertionError('Negative values detected in the CPT')
    # match all of the timesteps to the updated policy
    for tau in xrange(T0+1, T):
            G.basename_partition[bn][tau].CPT = G.basename_partition[bn][T0].CPT
    plt.plot(Rseries)
    plt.show()
    return G, Rseries