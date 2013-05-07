# -*- coding: utf-8 -*-
"""
Implements Optimistic Q-Learning for policies in pynfg.iterSemiNFG objects

Created on Fri Mar 22 15:32:33 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

Author: Dongping Xie

"""

from __future__ import division
import numpy as np
import matplotlib.pylab as plt

def q_to_cpt(qtable):
    """A function to convert a Q table to a CPT
    :arg qtable:
    
    """
    max_elems = qtable.max(qtable.ndim-1)
    step_1 = np.asarray(qtable.shape)
    step_1[-1] = 1
    max_elems = max_elems.reshape(step_1)
    dimension_multiplier = np.ones(qtable.ndim)
    dimension_multiplier[-1]=qtable.shape[-1]
    tiled_max = np.tile(max_elems, (dimension_multiplier))
    idx = qtable == tiled_max
    new_replaced = np.copy(qtable)
    num_max = np.sum(qtable==tiled_max, axis=(qtable.ndim - 1)).reshape(step_1)
    tiled_num_max = np.tile(num_max, (dimension_multiplier))
    new_replaced[idx] = 1
    cpt = (np.int_(new_replaced==1))/tiled_num_max
    return cpt
    
def opt_qlearning(G,bn,w,d,N,r_max = 0):
    """ Optimistic Q-learning: An Off-Policy TD Control RL Solution
    
    :arg G: The iterated semi-NFG on which to perform the RL
    :type G: iterSemiNFG
    :arg bn: The basename of the node with the CPT to be trained
    :type bn: str
    :arg w: The learning rate parameter
    :type w: float
    :arg d: The discount factor
    :type d: float
    :arg N: The number of training episodes
    :type N: int
    :arg r_max: (Optional) a guess of upperbound of reward in a single time 
        step. The default is 0 if no value is specified.
    :type r_max: float
    :returns: The iterated semi-NFG; a plot of the dynamic average reward; the
        q table
    
    Example::
        
        G1, rseries, Q1 = opt_qlearning(G,'D1',w=0.1,d=0.95,N=100):
    
    """
    T0 = G.starttime #get the start time
    T = G.endtime + 1 #get the end time
    player = G.bn_part[bn][T0].player #the player
    shape = G.bn_part[bn][T0].CPT.shape #the shape of CPT
    Q0 = r_max/(1-d) #the initial q value
    Q = Q0 * np.ones(shape) #the initial q table
    visit = np.zeros(shape) 
    #the number of times each (m,a) pair has been visited.                            
    r_av = 0 #the dynamic (discounted) average reward
    rseries = [] #a series of average rewards
    
    for ep in xrange(N):
        print ep
        G.bn_part[bn][T0].CPT = q_to_cpt(Q) #convert Q table to CPT
        G.sample_timesteps(T0,T0) #sample the start time step
        malist = G.bn_part[bn][T0].dict2list_vals(valueinput= \
                                                        G.bn_part[bn][T0].value)
        #get the list of (m,a) pair from the iterated semi-NFG
        mapair = G.bn_part[bn][T0].get_CPTindex(malist) #get CPT index
        
        r = G.reward(player,T0) #get the (discounted) reward

        if ep != 0: #to avoid "divided by 0" error
            r_av_new = r_av + (r-r_av)/((T-1)*ep) #update the dynamic reward

        Qmax = Q[mapair] #get the maximum q value
       
        for t in xrange(T0+1,T):
            G.bn_part[bn][t].CPT = q_to_cpt(Q) #convert Q table to CPT
            G.sample_timesteps(t,t) #sample the current time step
            if t!= (T-1): #required by Q-learning
                r = d**t*G.reward(player,t) # get the (discounted) reward
                r_av_new = r_av + (r-r_av)/((T-1)*ep+t) #update the reward

            malist_new = G.bn_part[bn][t].dict2list_vals(valueinput= \
                                                        G.bn_part[bn][t].value)
            mapair_new = G.bn_part[bn][t].get_CPTindex(malist_new)

            visit[mapair] = visit[mapair] + 1 #update the number of times
            alpha = (1/(1+visit[mapair]))**w #the learning rate
        
            Qmax_new = Q[mapair_new] #new maximum q value
            
            Q[mapair] = Qmax + alpha*(r + d*Qmax_new -Qmax) #update q table
            
            mapair = mapair_new 
            Qmax = Qmax_new
            r_av = r_av_new

        rseries.append(r_av)
    
    plt.plot(rseries) #plotting rseries to gauge convergence
    fig = plt.gcf() 
    plt.show()

    return G, fig
