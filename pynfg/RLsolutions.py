# -*- coding: utf-8 -*-
"""
Implements Reinforcement Learning solutions for iterSemiNFG objects

Part of: PyNFG - a Python package for modeling and solving Network Form Games

Created on Tue Jan 22 14:17:38 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""
from __future__ import division
import time
import numpy as np
import matplotlib.pylab as plt
from nodes import get_CPTindex, dict2list_vals

def ewma_mcrl(G, bn, J, N, alpha, delta, eps, uni=False, pureout=False):
    """ Use EWMA MC RL to approximate the optimal CPT at bn given G
    
    :arg G: The iterated semi-NFG on which to perform the RL
    :type G: iterSemiNFG
    :arg bn: the basename of the node with the CPT to be trained
    :type bn: str
    :arg J: The number of runs per training episode. If a schedule is desired, 
       enter a list or np.array with size equal to N.
    :type J: int, list, or np.array
    :arg N: The number of training episodes
    :type N: int
    :arg alpha: The exponential weight for the moving average. If a schedule is
       desired, enter a list or np.array with size equal to N
    :type alpha: int, list or np.array
    :arg delta: The discount factor 
    :type delta: float
    :arg eps: The maximum step-size for policy improvements
    :type eps: float
    
    Example::
        
        G1, Rseries = ewma_mcrl(G, 'D1', J=np.floor(linspace(300,100,num=50)), \
           N=50, alpha=1, delta=0.8, eps=0.4)
    
    """
    timepassed = np.zeros(N)
    # initializing training schedules from scalar inputs
    if isinstance(J, (int)):
        J = J*np.ones(N)
    if isinstance(alpha, (int, long, float)):
        alpha = alpha*np.ones(N)
    if isinstance(eps, (int, long, float)):
        eps = eps*np.ones(N)
    # getting shorter/more descriptive variable names to work with
    T0 = G.starttime
    T = G.endtime+1
    player = G.bn_part[bn][T0].player
    shape = G.bn_part[bn][T0].CPT.shape
    shape_last = shape[-1]
    if uni:
        G.bn_part[bn][T0].uniformCPT() #starting with a uniform CPT
    visit = set() #dict of the messages and mapairs visited throughout training
    R = 0 #average reward
    A = 0 #normalizing constant for average reward
    B = {} #dict associates messages and mapairs with beta exponents
    D = {} #dict associates messages and mapairs with norm constants for Q,V
    Q = np.zeros(shape) #Qtable
    V = np.zeros(shape[:-1]) #Value table
    Rseries = np.zeros(N) #tracking average reward for plotting convergence
    for n in xrange(N):
        print n
        indicaten = np.zeros(Q.shape) #indicates visited mapairs
        visitn = set() #dict of messages and mapairs visited in episode n
        Rseries[n] = R #adding the most recent ave reward to the data series
        A *= alpha[n] #rescaling A at start of new episode, see writeup
        for j in xrange(int(J[n])):
            visitj = set() #visitj must be cleared at the start of every run
            for t in xrange(T0,T):
                G.bn_part[bn][t].CPT = G.bn_part[bn][T0].CPT
                G.sample_timesteps(t, t) #sampling the timestep
                rew = G.reward(player, t) #getting the reward
                malist = dict2list_vals(G.bn_part[bn][t].parents, \
                                valueinput=G.bn_part[bn][t].value)
                # CPT index of messages and actions
                mapair = get_CPTindex(G.bn_part[bn][t], malist)
                # updating scalar dynamics
                a = A
                A = 1+a
                r = R
                R = (1/A)*(a*r+rew)
                xm = set() #used below to keep track of updated messages
                for values in visitj:
                    # past values
                    b = B[values]
                    d = D[values]
                    q = Q[values]
                    # update equations double letters are time t
                    bb = (b+1)
                    dd = d+1
                    qq = (1/dd)*(d*q+(delta**(bb-1))*(rew))    
                    # update dictionaries
                    B[values] = bb
                    D[values] = dd
                    Q[values] = qq
                    message = values[:-1] #V indexed by message only
                    if message not in xm: #updating message only once
                        # past values
                        b = B[message]
                        d = D[message]
                        v = V[message]
                        # update equations double letters are time t
                        bb = (b+1)
                        dd = d+1
                        vv = (1/dd)*(d*v+(delta**(bb-1))*(rew))    
                        # update dictionaries
                        B[message] = bb
                        D[message] = dd
                        V[message] = vv
                        xm.add(message) #so that message isn't updated again
                if mapair not in visitj: #first time in j visiting mapair
                    message = mapair[:-1]
                    messtrue = (message not in xm) #for checking message visited
                    B[mapair] = 1 #whenever mapair not in visitj
                    if mapair not in visitn and mapair not in visit:
                        D[mapair] = 1
                        Q[mapair] = rew
                        if messtrue:
                            D[message] = 1
                            V[message] = rew
                    elif mapair not in visitn:
                        D[mapair] = alpha[n]*D[mapair]+1
                        Q[mapair] = (1/D[mapair])*((D[mapair]-1)*Q[mapair]\
                                    +(rew))
                        if messtrue:
                            D[message] = alpha[n]*D[message]+1
                            V[message] = (1/D[message])*((D[message]-1)*\
                                        V[message]+(rew))
                    else:
                        D[mapair] += 1
                        Q[mapair] = (1/D[mapair])*((D[mapair]-1)*Q[mapair]\
                                    +(rew))
                        if messtrue:
                            D[message] += 1
                            V[message] = (1/D[message])*((D[message]-1)*\
                                        V[message]+(rew))
                    if messtrue:
                        B[message] = 1
                    # mapair gets added to visit sets the first time it appears
                    visit.add(mapair)
                    visitn.add(mapair)
                    visitj.add(mapair)
                    # only visited actions are updated 
                    indicaten[mapair] = 1
        go = time.time()
        # update CPT with shift towards Qtable argmax actions.
        shift = Q-V[...,np.newaxis]
        idx = np.nonzero(shift) # indices of nonzero shifts (avoid divide by 0)
        # normalizing shifts to be a % of message's biggest shift
        shiftnorm = np.absolute(shift).max(axis=-1)[...,np.newaxis]
        # for each mapair shift only eps% of the percent shift
        updater = eps[n]*indicaten*G.bn_part[bn][T0].CPT/shiftnorm
        # increment the CPT
        G.bn_part[bn][T0].CPT[idx] += updater[idx]*shift[idx]
        # normalize after the shift
        CPTsum = G.bn_part[bn][T0].CPT.sum(axis=-1)
        G.bn_part[bn][T0].CPT /= CPTsum[...,np.newaxis]
        timepassed[n] = time.time()-go
        if np.any(G.bn_part[bn][T0].CPT<0):
            raise AssertionError('Negative values detected in the CPT')
#    messages = set()
#    for mapair in visit:
#        if mapair[:-1] not in messages:
#            ind = G.bn_part[bn][T0].CPT[mapair[:-1],:].argmax()
#            G.bn_part[bn][T0].CPT[mapair[:-1],:] = 0
#            G.bn_part[bn][T0].CPT[mapair[:-1],ind]=1
#            messages.add(mapair[:-1])
    # before exiting, match all of the timesteps to the updated policy
    for tau in xrange(T0+1, T):
            G.bn_part[bn][tau].CPT = G.bn_part[bn][T0].CPT
    plt.plot(Rseries)
    plt.show()
    print np.mean(timepassed)
    return G, Rseries

#def ewma_jaakkola(G, bn, J, N, alpha, delta, eps):
#    """ Use EWMA MC to evaluate and improve the given Decision Node CPT
#    
#    :arg G: The iterated semi-NFG on which to perform the RL
#    :type G: iterSemiNFG
#    :arg bn: the basename of the CPT to be trained
#    :type bn: str
#    :arg J: The number of runs per training episode. If a schedule is desired, 
#       enter a list or np.array with size equal to N.
#    :type J: int, list, or np.array
#    :arg N: The number of training episodes
#    :type N: int
#    :arg alpha: The exponential weight for the moving average. If a schedule is
#       desired, enter a list or np.array with size equal to N
#    :type alpha: int, list or np.array
#    :arg delta: The discount factor 
#    :type delta: float
#    :arg eps: The maximum step-size for policy improvements
#    :type eps: float
#    """
#    # getting shorter/more descriptive variable names to work with
#    T0 = G.starttime
#    T = G.endtime+1
#    player = G.bn_part[bn][T0].player
#    shape = G.bn_part[bn][T0].CPT.shape
#    shape_last = shape[-1]
#    # starting with a uniform CPT
#    G.bn_part[bn][T0].uniformCPT()
#    # initializing RL parameters
#    if isinstance(J, (int)):
#        J = J*np.ones(N)
#    if isinstance(alpha, (int, long, float)):
#        alpha = alpha*np.ones(N)
#    if isinstance(eps, (int, long, float)):
#        eps = eps*np.ones(N)
#    visit = set()
#    R = 0
#    A = 0
#    B = {}
#    D = {}
#    Q = np.zeros(shape)
#    V = np.zeros(shape[:-1])
#    C = {}
#    I = {}
#    Rseries = np.zeros(N)
#    for n in xrange(N):
#        print n
#        # visitn must be cleared at the start of every episode
#        indicaten = np.zeros(Q.shape)
#        visitn = set()
#        Rseries[n] = R
#        A *= alpha[n]
#        for j in xrange(int(J[n])):
#            # visitj must be cleared at the start of every run
#            visitj = set()
#            for t in xrange(T0,T):
#                G.bn_part[bn][t].CPT = G.bn_part[bn][T0].CPT
#                G.sample_timesteps(t, t)
#                rew = G.reward(player, t)
##                parnodes = nodelist[t].parents.values()
#                malist = dict2list_vals(G.bn_part[bn][t].parents, \
#                                            valueinput=G.bn_part[bn][t].value)
#                mapair = get_CPTindex(G.bn_part[bn][t], malist)
#                # updating scalar dynamics
#                a = A
#                A = 1+a
#                r = R
#                R = (1/A)*(a*r+rew)
#                # updating set of visited (m,a) pairs
#                xm = set()
#                for values in visitj:
#                    # past values
#                    b = B[values]
#                    d = D[values]
#                    q = Q[values]
#                    c = C[values]
#                    # update equations double letters are time t
#                    bb = (b+1)
#                    dd = d+1
#                    qq = (1/dd)*(d*q+(delta**(bb-1))*(rew-r))    
#                    cc = (d/dd)*c+(1/dd)*(delta**(bb-1))
#                    qq += -cc*(R-r)
#                    # update dictionaries
#                    B[values] = bb
#                    D[values] = dd
#                    Q[values] = qq
#                    C[values] = cc
#                    # value function
#                    message = values[:-1]
#                    if message not in xm:
#                        # past values
#                        b = B[message]
#                        d = D[message]
#                        v = V[message]
#                        c = C[message]
#                        # current
#                        xj = True
#                        xn = True
#                        # update equations double letters are time t
#                        bb = (b+1)
#                        dd = d+1
#                        vv = (1/dd)*(d*v+(delta**(bb-1))*(rew-r))    
#                        cc = (d/dd)*c+(1/dd)*(delta**(bb-1))
#                        vv += -cc*(R-r)
#                        # update dictionaries
#                        B[message] = bb
#                        D[message] = dd
#                        V[message] = vv
#                        C[message] = cc
#                        xm.add(message)
#                if mapair not in visitj:
#                    message = mapair[:-1]
#                    messtrue = (message not in xm)
#                    B[mapair] = 1
#                    if mapair not in visitn and mapair not in visit:
#                        D[mapair] = 1
#                        Q[mapair] = rew-r
#                        C[mapair] = 1
#                        if messtrue:
#                            D[message] = 1
#                            V[message] = rew-r
#                            C[message] = 1
#                    elif mapair not in visitn:
#                        D[mapair] = alpha[n]*D[mapair]+1
#                        Q[mapair] = (1/D[mapair])*((D[mapair]-1)*Q[mapair]\
#                                    +(rew-r))
#                        C[mapair] = 1/D[mapair]
#                        if messtrue:
#                            D[message] = alpha[n]*D[message]+1
#                            V[message] = (1/D[message])*((D[message]-1)*\
#                                        V[message]+(rew-r))
#                            C[message] = 1/D[message]
#                    else:
#                        D[mapair] += 1
#                        Q[mapair] = (1/D[mapair])*((D[mapair]-1)*Q[mapair]\
#                                    +(rew-r))
#                        C[mapair] = ((D[mapair]-1)/D[mapair])*C[mapair]\
#                                    +(1/D[mapair])
#                        if messtrue:
#                            D[message] += 1
#                            V[message] = (1/D[message])*((D[message]-1)*\
#                                        V[message]+(rew-r))
#                            C[message] = ((D[message]-1)/D[message])*C[message]\
#                                            +(1/D[message])
#                    Q[mapair] += -C[mapair]*(R-r)
#                    if messtrue:
#                        B[message] = 1
#                        V[message] += -C[message]*(R-r)
#                    # mapair gets added to visit sets the first time it appears
#                    visit.add(mapair)
#                    visitn.add(mapair)
#                    indicaten[message] = 1
#                    visitj.add(mapair)
#        # update CPT with shift towards Qtable argmax actions.
#        shift = Q-V[...,np.newaxis]
#        idx = np.nonzero(shift)
#        shiftnorm = np.absolute(shift).max(axis=-1)[...,np.newaxis]
#        updater = eps[n]*indicaten*G.bn_part[bn][T0].CPT/shiftnorm
#        G.bn_part[bn][T0].CPT[idx] += updater[idx]*(Q-V[...,np.newaxis])[idx]
#        CPTsum = G.bn_part[bn][T0].CPT.sum(axis=-1)
#        G.bn_part[bn][T0].CPT = G.bn_part[bn][T0].CPT/CPTsum[...,np.newaxis]
#        if np.any(G.bn_part[bn][T0].CPT<0):
#            raise AssertionError('Negative values detected in the CPT')
#    # match all of the timesteps to the updated policy
#    for tau in xrange(T0+1, T):
#            G.bn_part[bn][tau].CPT = G.bn_part[bn][T0].CPT
#    plt.plot(Rseries)
#    plt.show()
##    print 'J: %s' %J
##    print 'alpha: %s' %alpha
##    print 'eps: %s' %eps
##    print 'delta: %s' %delta
#    return G, Rseries