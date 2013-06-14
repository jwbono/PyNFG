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
from pynfg.utilities.utilities import convert_2_pureCPT, iterated_input_dict
import copy
import warnings

class QLearning(object):
    """
    Finds the **uncoordinated** best policy using Q-learning.
    :arg Game: The iterated semi-NFG on which to perform the RL
    :type Game: iterSemiNFG
    :arg specs: A nested dictionary contained specifications of the
        game.  See below for details
    :type specs: dict

    The specs dictionary is a triply nested dictionary.  The first
    level of keys is player names.  For each player there is an entry for
    the player's:

    Level : int
        The player's level
    w : float
       The learning rate
    delta : float
        The discount factor

    The rest of the entries are basenames.  The value of
    each basename is a dictionary containing:

    N : int
        The number of training episodes
    r_max : float
        (Optional) a guess of upperbound of reward in a single time
            step. The default is 0 if no value is specified.

    """
    def __init__(self, G, specs):
        self.G = copy.deepcopy(G)
        self.specs = specs
        self.trained_CPTs = {}
        self.figs = {}
        for player in G.players:
            basenames = set(map(lambda x: x.basename, G.partition[player]))
            for bn in basenames:
                self.trained_CPTs[player] = {}
                self.trained_CPTs[player][bn] = {}
                self.trained_CPTs[player][bn]['Level0'] = self._set_L0_CPT()
                self.figs[bn] = {}
        self.high_level = max(map(lambda x: self.specs[x]['Level'], G.players))

    def _set_L0_CPT(self):
        """ Sets the level 0 CPT"""
        G = self.G
        ps = self.specs
        for player in ps:
            basenames = set(map(lambda x: x.basename, G.partition[player]))
            for bn in basenames:
                if ps[player][bn]['L0Dist'] == 'uniform':
                    return G.bn_part[bn][0].uniformCPT(setCPT=False)
                elif ps[player][bn]['L0Dist'] is None:
                    warnings.warn("No entry for L0Dist for player %s,\
                    setting to current CPT" % player)
                    return G.bn_part[bn][0].CPT
                elif type(ps[player][bn]['L0Dist']) == np.ndarray:
                    return ps[player][bn]

    def train_node(self, bn, level, setCPT=False):
        """Solve for the optimal policy using Optimistic Q-learning

        Optimistic Q-Learning  is an off-policy TD control RL algorithm

        :arg bn: The basename of the node with the CPT to be trained
        :type bn: str
        :arg level: The level at which to train the basename
        :type level: int
        """
        print 'Training ' + bn + ' at level '+ str(level)
        G = copy.deepcopy(self.G)
        ps = self.specs
        player = G.bn_part[bn][0].player
        w, d, N, r_max = ps[player]['w'], ps[player]['delta'], ps[player][bn]['N'], \
            ps[player][bn]['r_max']
        #Set other CPTs to level-1.  Works even if CPTs aren't pointers.
        for o_player in G.players:
            bn_list = list(set(map(lambda x: x.basename, G.partition[o_player])))
            for base in bn_list:
                if base != bn:
                    for dn in G.bn_part[base]:
                        try:
                            dn.CPT = \
                                (self.trained_CPTs[o_player][base]['Level' +
                                                          str(level - 1)])
                        except KeyError:
                            raise KeyError('Need to train other players at level %s'
                                   % str(level-1))
        T0 = G.starttime #get the start time
        T = G.endtime + 1 #get the end time
        shape = G.bn_part[bn][T0].CPT.shape #the shape of CPT
        if d<1:
            Q0 = r_max*((1-d**(T-T0))/(1-d)) #the initial q value
        else:
            Q0 = r_max*(T-T0)
        Q = Q0 * np.ones(shape) #the initial q table
        visit = np.zeros(shape)
        #the number of times each (m,a) pair has been visited.
        r_av = 0 #the dynamic (discounted) average reward
        rseries = [] #a series of average rewards
        for ep in xrange(N):
            print ep
            #convert Q table to CPT
            G.bn_part[bn][T0].CPT = convert_2_pureCPT(Q)
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
                G.bn_part[bn][t].CPT = convert_2_pureCPT(Q) #convert Q table to CPT
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
        self.trained_CPTs[player][bn]['Level' + str(level)] = G.bn_part[bn][0].CPT
        plt.figure()
        plt.plot(rseries, label = str(bn + ' Level ' + str(level)))
        #plotting rseries to gauge convergence
        plt.legend()
        fig = plt.gcf()
        self.figs[bn][str(level)] = fig
        if setCPT:
            map(lambda x: _setallCPTs(self.G,bn, x, G.bn_part[bn][0].CPT), np.arange(T0, T))


    def solve_game(self, setCPT=False):
        """Solves the game sfor specified player levels"""
        G = self.G
        ps = self.specs
        for level in np.arange(1, self.high_level):
            for player in G.players:
                basenames = set(map(lambda x: x.basename, G.partition[player]))
                for controlled in basenames:
                    self.train_node(controlled, level, setCPT=setCPT)
        for player in G.players:
            basenames = set(map(lambda x: x.basename, G.partition[player]))
            for controlled in basenames:
                if ps[player]['Level'] == self.high_level:
                    self.train_node(controlled, self.high_level, setCPT=setCPT)


def qlearning_dict(G, Level, w, N, delta, r_max=0, L0Dist=None):
    """
    Creates the specs shell for a game to be solved using Q learning.

    :arg G: An iterated SemiNFG
    :type G: iterSemiNFG

    .. seealso::
        See the Q Learning documentation (above) for details of the  optional arguments
    """
    return iterated_input_dict(G, [('Level', Level), ('delta', delta),('w', w)],
                                  [('L0Dist', L0Dist), ('N', N),
                                   ('N', N), ('r_max', r_max)])

def _setallCPTs(G,basename, t, newCPT):
    G.bn_part[basename][t].CPT = newCPT
