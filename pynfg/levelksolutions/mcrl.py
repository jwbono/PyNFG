# -*- coding: utf-8 -*-
"""
Implements Monte Carlo Reinforcement Learning for iterSemiNFG objects

Created on Mon Feb 18 09:03:32 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""
from __future__ import division
import time
import copy
import numpy as np
import matplotlib.pylab as plt
from pynfg.utilities.utilities import iterated_input_dict
import warnings
import sys

class EWMA_MCRL(object):
    """
    Finds the **uncoordinated** best policy using reinforcement learning.

    :arg Game: The iterated semi-NFG on which to perform the RL
    :type Game: iterSemiNFG
    :arg specs: A nested dictionary containing specifications of the
        game.  See below for details
    :type specs: dict

    The specs dictionary is a triply nested dictionary.  The first
    level of keys is player names.  For each player there is an entry with key

    Level : int
        The player's level

    The rest of the entries are basenames.  The value of each basename is a
    dictionary containing:

    J : int, list, or np.array
        The number of runs per training episode. If a schedule is desired, enter a list or np.array with size equal to N.
    N : int
         The number of training episodes
    L0Dist : ndarray, str, None
        If ndarray, then the level 0 CPT is set to
        L0Dist. If L0Dist is 'uniform', then all Level 0 CPTs are set to
        the uniform distribution.  If L0Dist is None, then the level 0 CPT
        is set to the CPT of the inputted game.
    alpha : int, list or np.array
        The exponential weight for the moving average. If a schedule is
        desired, enter a list or np.array with size equal to N
    delta : float
        The discount factor
    eps : float
        The maximum step-size for policy improvements
    pureout : bool
        if True, the policy is turned into a pure policy at the end
        of training by assigning argmax actions prob 1. Default is False

    """
    def __init__(self, Game, specs):
        self.Game = copy.deepcopy(Game)
        self.specs = specs
        self.trained_CPTs = {}
        self.figs = {}
        for player in Game.players:
            basenames = set(map(lambda x: x.basename, Game.partition[player]))
            for bn in basenames:
                self.figs[bn]={}
                self.trained_CPTs[player] = {}
                self.trained_CPTs[player][bn] = {}
                self.trained_CPTs[player][bn][0] = self._set_L0_CPT()
        self.high_level = max(map(lambda x: self.specs[x]['Level'], Game.players))

    def _set_L0_CPT(self):
        """ Sets the level 0 CPT"""
        Game = self.Game
        ps = self.specs
        for player in ps:
            basenames = set(map(lambda x: x.basename, Game.partition[player]))
            for bn in basenames:
                if ps[player][bn]['L0Dist'] == 'uniform':
                    return Game.bn_part[bn][0].uniformCPT(setCPT=False)
                elif ps[player][bn]['L0Dist'] is None:
                    warnings.warn("No entry for L0Dist for player %s,\
                    setting to current CPT" % player)
                    return Game.bn_part[bn][0].CPT
                elif type(ps[player][bn]['L0Dist']) == np.ndarray:
                    return ps[player][bn]

    def train_node(self, bn, level, setCPT=False):
        """ Use EWMA MC RL to approximate the optimal CPT at bn given Game

        :arg bn: the basename of the node with the CPT to be trained
        :type bn: str
        :arg level: The level at which to train the basename
        :type level: int
        """
        sys.stdout.write('\r')
        print 'Training ' + bn + ' at level '+ str(level)
        specs = self.specs
        Game = copy.deepcopy(self.Game)
        player = Game.bn_part[bn][0].player
        basedict = specs[player][bn]
        J, N, alpha, delta, eps, pureout = basedict['J'], basedict['N'], \
            basedict['alpha'], basedict['delta'], basedict['eps'], \
            basedict['pureout']
        #Set other CPTs to level-1.  Works even if CPTs aren't pointers.
        for o_player in Game.players:
            bn_list = list(set(map(lambda x: x.basename, Game.partition[o_player])))
            for base in bn_list:
                if base != bn:
                    for dn in Game.bn_part[base]:
                        try:
                            dn.CPT = \
                                (self.trained_CPTs[o_player][base][level - 1])
                        except KeyError:
                            raise KeyError('Need to train other players at level %s'
                                   % str(level-1))
        # initializing training schedules from scalar inputs
        if isinstance(J, (int)):
            J = J*np.ones(N)
        if isinstance(alpha, (int, long, float)):
            alpha = alpha*np.ones(N)
        if isinstance(eps, (int, long, float)):
            eps = eps*np.ones(N)
        # getting shorter/more descriptive variable names to work with
        T0 = Game.starttime
        T = Game.endtime+1
        shape = Game.bn_part[bn][0].CPT.shape
        shape_last = shape[-1]
        for dn in Game.bn_part[bn]:  # pointing all CPTs to T0, i.e. single policy
            dn.CPT = Game.bn_part[bn][0].CPT
        visit = set()  # dict of the messages and mapairs visited throughout training
        R = 0  # average reward with initial value of zero
        A = 0  # normalizing constant for average reward
        B = {}  # dict associates messages and mapairs with beta exponents
        D = {}  # dict associates messages and mapairs with norm constants for Q,V
        Q = np.zeros(shape)  # Qtable
        V = np.zeros(shape[:-1])  # Value table
        Rseries = np.zeros(N)  # tracking average reward for plotting convergence
        np.seterr(invalid='ignore', divide='ignore')
        for n in xrange(N):
            sys.stdout.write('\r')
            sys.stdout.write('Iteration ' + str(n))
            sys.stdout.flush()
            indicaten = np.zeros(Q.shape)  # indicates visited mapairs
            visitn = set()  # dict of messages and mapairs visited in episode n
            Rseries[n] = R  # adding the most recent ave reward to the data series
            A *= alpha[n]  # rescaling A at start of new episode, see writeup
            for j in xrange(int(J[n])):
                visitj = set()  # visitj must be cleared at the start of every run
                for t in xrange(T0, T):
                    #import pdb; pdb.set_trace()
                    #Game.bn_part[bn][t-T0].CPT = copy.copy(Game.bn_part[bn][0].CPT)
                    Game.sample_timesteps(t, t)  # sampling the timestep
                    rew = Game.reward(player, t)  # getting the reward
                    mapair = Game.bn_part[bn][t-T0].get_CPTindex()
                    A += 1
                    r = R
                    R = (1/A)*((A-1)*r+rew)
                    xm = set()  # used below to keep track of updated messages
                    for values in visitj:
                        b = B[values]  # past values
                        d = D[values]
                        q = Q[values]
                        bb = (b+1)  # update equations double letters are time t
                        dd = d+1
                        qq = (1/dd)*(d*q+(delta**(bb-1))*(rew))
                        B[values] = bb  # update dictionaries
                        D[values] = dd
                        Q[values] = qq
                        message = values[:-1]  # V indexed by message only
                        if message not in xm:  # updating message only once
                            b = B[message]  # past values
                            d = D[message]
                            v = V[message]
                            bb = (b+1)  # update equations double letters are time t
                            dd = d+1
                            vv = (1/dd)*(d*v+(delta**(bb-1))*(rew))
                            B[message] = bb  # update dictionaries
                            D[message] = dd
                            V[message] = vv
                            xm.add(message)  # so that message isn't updated again
                    if mapair not in visitj:  # first time in j visiting mapair
                        message = mapair[:-1]
                        messtrue = (message not in xm)  # for checking message visited
                        B[mapair] = 1  # whenever mapair not in visitj
                        if mapair not in visitn and mapair not in visit:
                            D[mapair] = 1
                            Q[mapair] = rew
                            if messtrue:
                                D[message] = 1
                                V[message] = rew
                        elif mapair not in visitn:
                            D[mapair] = alpha[n]*D[mapair]+1
                            Q[mapair] = (1/D[mapair])*((D[mapair]-1)*Q[mapair]
                                       +(rew))
                            if messtrue:
                                D[message] = alpha[n]*D[message]+1
                                V[message] = (1/D[message])*((D[message]-1)*\
                                            V[message]+(rew))
                        else:
                            D[mapair] += 1
                            Q[mapair] = (1/D[mapair])*((D[mapair]-1)*Q[mapair]\
                                    + (rew))
                            if messtrue:
                                D[message] += 1
                                V[message] = (1/D[message])*((D[message]-1) *
                                             V[message]+(rew))
                        if messtrue:
                            B[message] = 1
                        visit.add(mapair)  # mapair added to visit sets the first time
                        visitn.add(mapair)
                        visitj.add(mapair)
                        indicaten[mapair] = 1  # only visited actions are updated
            #  update CPT with shift towards Qtable argmax actions.
            shift = Q-V[...,np.newaxis]
            idx = np.nonzero(shift)  # indices of nonzero shifts (avoid divide by 0)
            # normalizing shifts to be a % of message's biggest shift
            shiftnorm = np.absolute(shift).max(axis=-1)[...,np.newaxis]
            # for each mapair shift only eps% of the percent shift
            updater = eps[n]*indicaten*Game.bn_part[bn][0].CPT/shiftnorm
            # increment the CPT
            Game.bn_part[bn][0].CPT[idx] += updater[idx]*shift[idx]
            # normalize after the shift
            CPTsum = Game.bn_part[bn][0].CPT.sum(axis=-1)
            Game.bn_part[bn][0].CPT /= CPTsum[...,np.newaxis]
        if pureout: #if True, output is a pure policy
            Game.bn_part[bn][0].makeCPTpure()
        self.trained_CPTs[player][bn][level] = Game.bn_part[bn][0].CPT
        if setCPT:
            for node in self.Game.bn_part[bn]:
                node.CPT = Game.bn_part[bn][0].CPT
        for tau in xrange(1, T-T0): #before exit, make CPTs independent in memory
            Game.bn_part[bn][tau].CPT = copy.copy(Game.bn_part[bn][0].CPT)
        plt.figure()
        plt.plot(Rseries, label = str(bn + ' Level ' + str(level)))
        #plotting rseries to gauge convergence
        plt.legend()
        fig = plt.gcf()
        self.figs[bn][str(level)] = fig
        sys.stdout.write('\n')

    def solve_game(self, setCPT=False):
        """Solves the game for given player levels"""
        Game = self.Game
        ps = self.specs
        for level in np.arange(1, self.high_level):
            for player in Game.players:
                basenames = set(map(lambda x: x.basename, Game.partition[player]))
                for controlled in basenames:
                    self.train_node(controlled, level, setCPT=setCPT)
        for player in Game.players:
            basenames = set(map(lambda x: x.basename, Game.partition[player]))
            for controlled in basenames:
                if ps[player]['Level'] == self.high_level:
                    self.train_node(controlled, self.high_level, setCPT=setCPT)


def mcrl_dict(Game, Level, J, N, delta, alpha=.5, eps=.2, L0Dist=None,
               pureout=False):
    """
    Creates the specs shell for a game to be solved using MCRL.

    :arg Game: An iterated SemiNFG
    :type Game: SemiNFG

    .. seealso::
        See the EWMA_MCRL documentation (above) for details of the  optional arguments

    """
    return iterated_input_dict(Game, [('Level', Level)], [('L0Dist', L0Dist), ('J', J),
                                                      ('N', N), ('delta', delta),
                                                      ('alpha', alpha), ('eps', eps),
                                                      ('pureout', pureout)])
