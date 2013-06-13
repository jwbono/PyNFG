# -*- coding: utf-8 -*-
"""
Implements Best Response Level-K calculations for SemiNFG and iterSemiNFG

Part of: PyNFG - a Python package for modeling and solving Network Form Games

Created on Fri May 24 07:01:05 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""
from __future__ import division

import copy
import numpy as np
from pynfg.utilities.utilities import convert_2_pureCPT, mceu, input_dict, iterated_input_dict
import warnings
import pynfg


class BestResponse(object):
    """ Finds the best solution for a semi-NFG.

    :arg G:  A semi-NFG
    :type G: semiNFG
    :arg specs: dictionary of dictionaries containing specifications
        the level, level-0 strategy, tolerance
        and degrees of rationality of each player.
        See below for details.
    :type specs: dict
    :arg N: Number of times to repeat sampling algorithm
    :type N: int

    specs is a triply-nested dictionary.  The first set of keys
    are the player names.  For each player key, there are keys:

    Level : int
        The player's Level
    delta : float
        The discount factor

    The rest of the keys for each player are the names of nodes that belong to that
    player.  For each node, the dictionary has three entries with one optional entry:


    L0Dist : ndarray, str, None
        If ndarray, then the level 0 CPT is set to
        L0Dist. If L0Dist is 'uniform', then all Level 0 CPTs are set to
        the uniform distribution.  If L0Dist is None, then the level 0 CPT
        is set to the CPT of the inputted game.
    tol : int
        the minimum number of samples per parent value
    N : int
        The max number of iterations for the estimation.

    beta : float
        (Optional)  Logit best response parameter

    """

    def __init__(self, G, specs, logit=False):
        self.G = copy.deepcopy(G)
        self.logit = logit
        self.specs = specs
        # if type(G) == pynfg.classes.iterseminfg.iterSemiNFG:
        #     self.iterated = True
        #     self.trained_CPTs = {}
        #     for player in G.players:
        #         basenames = set(map(lambda x: x.basename, G.partition[player]))
        #         for bn in basenames:
        #             self.trained_CPTs[player] = {}
        #             self.trained_CPTs[player][bn] = {}
        #             self.trained_CPTs[player][bn]['Level0'] = self._set_iter_L0_CPT()
        #             self.high_level = max(map(lambda x: self.specs[x]['Level'], G.players))
        # else:
        #     self.iterated = False
        self.high_level = self._set_new_attributes()
        self._set_L0_CPT()


    def _set_new_attributes(self):
        G = self.G
        ps = self.specs
        levels = []
        for player in ps:
            node_set = list(G.partition[player])
            for node in node_set:
                nodename = node.name
                node.Level, node.delta, node.tol, node.N =  \
                    ps[player]['Level'], ps[player]['delta'],\
                    ps[player][nodename]['tol'], ps[player][nodename]['N']
                if self.logit:
                    node.beta = ps[player][nodename]['beta']
                try:
                    node.LevelCPT
                except AttributeError:
                    node.LevelCPT = {}
            levels.append(ps[player]['Level'])
        return max(levels)

    def _set_L0_CPT(self):
        """ Sets the level 0 CPT"""
        G = self.G
        ps = self.specs
        for player in ps:
            node_set = list(G.partition[player])
            for node in node_set:
                try:
                    node.LevelCPT['Level0']
                except KeyError:
                    nodename = node.name
                    if ps[player][nodename]['L0Dist'] == 'uniform':
                        node.LevelCPT['Level0'] = \
                            node.uniformCPT(setCPT=False)
                    elif ps[player][nodename]['L0Dist'] is None:
                        warnings.warn("No entry for L0Dist for player %s,\
                        setting to current CPT" % player)
                        node.LevelCPT['Level0'] = G.node_dict[nodename].CPT
                    elif type(ps[player][nodename]['L0Dist']) == np.ndarray:
                        node.LevelCPT['Level0'] = \
                            ps[player][nodename]['L0Dist']

    def train_node(self, nodename, level, setCPT=False, verbose=False):
        """Compute level-k best response at the DN given Game

        :arg nodename: the name of the decision node where MCEUs are estimated
        :type nodename: str
        :arg level: The level at which to train that player
        :type level: int
        :arg setCPT: If the trained CPT should be set as the current CPT.
            Otherwise, it can be accessed through node.LevelCPT.  Default is
            False
        :type setCPT: bool

        """
        print 'Training ' + nodename + ' at level ' + str(level)
        G = copy.deepcopy(self.G)  # copy in order to maintain original CPT
        ps = self.specs
        for node in G.node_dict:  # G changes, self.G doesn't
            if type(node) is pynfg.classes.decisionnode.DecisionNode:
                try:
                    node.CPT = node.LevelCPT['Level' + str(level - 1)]
                except KeyError:
                    raise KeyError('Need to train other players at level %s'
                                   % str(level-1))
        EUtable = mceu(G, nodename, G.node_dict[nodename].N,
                       G.node_dict[nodename].tol, G.node_dict[nodename].delta,
                       verbose=verbose)
        if not self.logit:
            self.G.node_dict[nodename].LevelCPT['Level' + str(level)] = \
                  convert_2_pureCPT(EUtable)
            if setCPT:
                self.G.node_dict[nodename].CPT = convert_2_pureCPT(EUtable)
        else:
            weight = np.exp(G.node_dict[nodename].beta*EUtable)
            norm = np.sum(weight, axis=-1)
            self.G.node_dict[nodename].LevelCPT['Level' + str(level)] = \
            weight/norm[..., np.newaxis]
            if setCPT:
                self.G.node_dict[nodename].CPT = weight/norm[..., np.newaxis]

        # else:
        #     player = G.node_dict[nodename].player
        #     for o_player in G.players:
        #         bn_list = list(set(map(lambda x: x.basename, G.partition[o_player])))
        #         for base in bn_list:
        #             if base != bn:
        #                 try:
        #                     G.bn_part[base][0].CPT = \
        #                         self.trained_CPTs[o_player][base]['Level' +
        #                                                   str(level - 1)]
        #                 except KeyError:
        #                     raise KeyError('Need to train other players at level %s'
        #                            % str(level-1))
        #     EUtable = mceu(G, G.bn_part[bn][0].name, ps[player][bn]['N'],
        #                    ps[player][bn]['tol'], ps[player]['delta'],
        #                    verbose=verbose)


    def solve_game(self, setCPT=False, verbose=False):
        """ Solves the game for specified player levels"""
        G = self.G
        for level in np.arange(1, self.high_level):
            for player in G.players:
                for controlled in G.partition[player]:
                    self.train_node(controlled.name, level, verbose=verbose)
        for player in G.players:
            for controlled in G.partition[player]:
                if controlled.Level == self.high_level:
                    self.train_node(controlled.name, self.high_level,
                                    verbose=verbose)
        if setCPT:
            for player in G.players:
                for node in G.partition[player]:
                    G.node_dict[node.name].CPT = G.node_dict[node.name].\
                        LevelCPT['Level' + str(G.node_dict[node.name].Level)]


def br_dict(G, N, Level, L0Dist=None, delta=.1, tol=30, beta=None):
    """A helper function to generate the player_spec dictionary
    for relaxed level K.  If optional arguments are specified, they are
    set for all decision nodes.

    :arg G: A SemiNFG
    :type G: SemiNFG

    .. seealso::
        See the BestResponse documentation (above) for details of the  optional arguments
    """
    # if type(G) is pynfg.classes.iterseminfg.iterSemiNFG:
    #     iterated =True
    # if not iterated:
    if beta is None:
        return input_dict(G, [('Level', Level), ('delta', delta)],
                          [('L0Dist', L0Dist), ('N', N), ('tol', tol)])
    else:
        return input_dict(G, [('Level', Level), ('delta', delta)],
                      [('L0Dist', L0Dist), ('N', N), ('tol', tol),
                       ('beta', beta)])
    # else:
    #     if beta is None:
    #          return iterated_input_dict(G, [('Level', Level), ('delta', delta)],
    #                       [('L0Dist', L0Dist), ('N', N), ('tol', tol)])
    #     else:
    #         return iterated_input_dict(G, [('Level', Level), ('delta', delta)],
    #                   [('L0Dist', L0Dist), ('N', N), ('tol', tol),
    #                    ('beta', beta)])
