import copy as copy
import warnings
import numpy as np
import itertools


class rlk(object):
    """ Finds the relaxed level-k solution for a semi-NFG.

    References
    ----------
    Lee and Wolpert, "Game theoretic modeling of pilot behavior
    during mid-air encounters," Decision-Making with Imperfect
    Decision Makers, T. Guy, M. Karny and D.H.Wolpert,
    Springer (2011).

    :arg G:  A semi-NFG
    :type G: semiNFG
    :arg player_specs: dictionary of dictionaries containing
    the level, satisficing distribution, level-0 strategy,
    and degrees of rationality of each player.
    See below for details.
    :type player_specs: dict

    player_spec is a triply-nested dictionary.  The first set of keys
    are the player names.  For each player key, there is a dictionary
    whose keys are the names of nodes that belong to that player.
    For each node, the dictionary has five entries:

    Level : int
        The level of the player
    M : int
        The number of times to sample the satisficing distribution
    Mprime : int
        The number of times to sample the net for each satisficing
        draw.
    L0Dist : ndarray, str, None
        If ndarray, then the level 0 CPT is set to
        L0Dist. If L0Dist is 'uniform', then all Level 0 CPTs are set to
        the uniform distribution.  If L0Dist is None, then the level 0 CPT
        is set to the CPT of the inputted game.
    SDist : function, 2darray, or  str
        If 'all pure' then the satisficing distribution is all
        pure strategies.  If 'all mixed' then the satisficing
        distribution is all mixed strategies.  If 2darray,
        the value must be an nxk array where n is the number
        of satisficing strategies and k is the size of the
        player's space.  Each row in the 2darray corresponds to
        a strategy.  If function, SDist is a function that returns
        **a draw** from the conditional satisficing distribution.
        The function can take as parameters a dictionary whose keys
        are the names of the parent nodes and values are the value
        of the parent node.

    """
    def __init__(self, G, player_specs, N, parallel=False):
        self.parallel = parallel
        self.player_specs = player_specs
        self.N = N
        if not parallel:
            self.G = copy.deepcopy(G)
            self.high_level = self._set_new_attributes()  # also sets attributes
            self._set_L0_CPT()
            self._set_satisficing_func()
        if parallel:
            self.G = G
            try:
                self.high_level
            except AttributeError:
                self.high_level = self._set_new_attributes()
                self._set_L0_CPT()
                self._set_satisficing_func()

    def _set_new_attributes(self):
        """ Sets the level and rationality of a decision node

        It assigns an attribute "Level", "M" and "Mprime" to each decision
        node.
        """
        G = self.G
        ps = self.player_specs
        levels = []
        for player in ps:
            node_set = list(G.partition[player])
            for node in node_set:
                nodename = node.name
                node.Level, node.M, node.Mprime =  \
                    ps[player][nodename]['Level'], ps[player][nodename]['M'],\
                    ps[player][nodename]['Mprime']
                try:
                    node.LevelCPT
                except AttributeError:
                    node.LevelCPT = {}
            levels.append(ps[player][nodename]['Level'])
        return max(levels)

    def _set_L0_CPT(self):
        """ Sets the level 0 CPT"""
        G = self.G
        ps = self.player_specs
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

    def _set_satisficing_func(self):
        """Creates bound method that draws from satisficing distribution"""
        G = self.G
        ps = self.player_specs
        for player in ps:
            nodeset = G.partition[player]
            for node in nodeset:
                nodename = node.name
                sd = ps[player][nodename]['SDist']
                if hasattr(sd, '__call__'):  # If a function
                    node.SDist = sd
                elif type(sd) == np.ndarray:
                    node.SDist = self.sfunc(node, 'arr', sd)
                elif sd == 'all pure':
                    node.SDist = self.sfunc(node, 'all pure')
                elif sd == 'all mixed':
                    node.SDist = self.sfunc(node, 'all mixed')

    def _draw_from_array(self, nd,  ndar):
        """ A draw from a satisficing distribution"""
        s0shape = len(ndar.shape)
        if s0shape[1] != len(nd.space):
            raise ValueError('ndarray second dimension needs be \
            the same as the number of elements in the player\'s space')
        line = np.random.randint(0, s0shape[0])
        return ndar[line]

    def _draw_all_pure(self, nd):
        """ A draw from a satisficing distribution of all pure strategies"""
        s0shape = len(nd.space)
        strat = np.zeros(s0shape)
        strat[np.random.randint(0, s0shape)] = 1
        return strat

    def _draw_all_mixed(self, nd):
        """ A draw from a satisficing distribution of all mixed strategies"""
        s0shape = len(nd.space)
        strat = np.random.dirichlet(np.ones(s0shape))
        return strat

    def sfunc(self, nd, form, ndar=None):
        """Wrapper to draw from satisficing distribution """
        def sgen(*args):
            if form == 'arr':
                return self._draw_from_array(nd, ndar)
            if form == 'all pure':
                return self._draw_all_pure(nd)
            if form == 'all mixed':
                return self._draw_all_mixed(nd)
        return sgen

    def sample_CPT(self, nodename, level):
        """ Samples entire CPT according to Deifnition 7 in Lee and Wolpert"""
        G = self.G
        node = G.node_dict[nodename]
        other_level = 'Level%s' % str(level-1)
        for player in G.players:  # Sets all players to lower level
            if player != node.player:
                try:
                    for controlled in G.partition[player]:
                        controlled.CPT = \
                            np.copy(controlled.LevelCPT[other_level])
                except KeyError:
                    raise KeyError('Need to train other players at level %s'
                                   % str(level-1))
        Y = copy.copy(G.node_dict)  # Create Y
        Y.pop(node.name)
        [Y.pop(pa) for pa in node.parents.keys()]
        [Y.pop(suc.name) for suc in G.descendants(node.name)]
        parent_space = []
        for par in node.parents.values():
            parent_space.append(par.space)
            parent_combs = itertools.product(*parent_space)  # Iterate pa(node)
        trained_CPT = np.zeros(node.CPT.shape)
        for combo in parent_combs:  # For each parent
            ix = []
            p_node_val = zip(node.parents.keys(), combo)
            for elem in p_node_val:
                ix.append(G.node_dict[elem[0]].space.index(elem[1]))
                ix = tuple(ix)  # Used to set CPT 'row' to draw value
            max_util = - np.inf
            G.set_values(dict(p_node_val))  # Sets parents
            Y_vals = self.sample_set(Y.keys(), node.Mprime)     # STEP 2
            satis_set = []
            for m in range(node.M):  # STEP 1
                sdist = node.SDist()
                if list(sdist) not in satis_set:
                    satis_set.append((list(sdist)))

            for sdraw in satis_set:  # For each SDist
                node.CPT[ix] = sdraw
                node.draw_value()  # set CPT and draw
                weu = []
                for y in Y_vals:
                    G.set_values(y)             # Step 2 B (below)
                    wt = np.prod([n.prob() for n in node.parents.values()])
                    succ_samp = self.sample_set([n.name
                                                for n in G.descendants(
                                                    node.name)], 1)[0]
                    G.set_values(succ_samp)  # STEP 3
                    weu.append(wt * G.utility(node.player))
                EU = np.mean(weu)
                if EU >= max_util:
                    if EU > max_util:
                        num_best = 1
                        max_util = EU
                        best_strat = sdraw
                    if EU == max_util:
                        best_strat = np.asarray(best_strat) * float(num_best) /\
                            float(num_best + 1) + \
                            np.asarray(sdraw)/float(num_best + 1)
                        num_best += 1  # Mean update (above)
            trained_CPT[ix] = best_strat
        return trained_CPT

    def sample_set(self, nodenames, Mprime):
        """ Returns a list with length Mprime
        whose elements are a dictionary of samples of nodes.
        """
        G = copy.deepcopy(self.G)
        set_dicts = []
        for i in range(Mprime):
            set_samp = {}
            for n in G.iterator:
                if n.name in nodenames:
                    set_samp[n.name] = n.draw_value(setvalue=False)
            set_dicts.append(set_samp)

        return set_dicts

    def train_node(self, nodename, level, setCPT=False):
        G = self.G
        print "Training " + nodename + " at level " + str(level)
        node = G.node_dict[nodename]
        CPT = np.zeros(node.CPT.shape)
        for mcsamp in xrange(self.N):
            new_CPT = self.sample_CPT(nodename, level)
            CPT = CPT * float(mcsamp)/float(mcsamp+1) + \
                new_CPT / float(mcsamp + 1)
        Levelkey = 'Level' + str(level)
        node.LevelCPT[Levelkey] = CPT
        if setCPT:
            node.CPT = CPT

    def solve_game(self, setCPT=False):
        G = self.G
        for level in np.arange(1, self.high_level):
            for player in G.players:
                for controlled in G.partition[player]:
                    self.train_node(controlled.name, level)
        for player in G.players:
            for controlled in G.partition[player]:
                if controlled.Level == self.high_level:
                    self.train_node(controlled.name, self.high_level)
        if setCPT:
            for player in G.players:
                for node in G.partition[player]:
                    G.node_dict[node.name].CPT = G.node_dict[node.name].LevelCPT['Level' +
                                                    str(G.node_dict[node.name].Level)]

def rlk_dict(G, M=None, Mprime=None, Level=None, L0Dist=None, SDist=None):
    """ A helper function to generate the player_spec dictionary
    for relaxed level K.  If optional arguments are specified, they are
    set for all decision nodes.

    :arg G: A SemiNFG
    :type G: SemiNFG

    .. seealso::
        See the rlk (TODO: Sphinx link)  for definition of optional arguments
    """
    rlk_input = {}
    for player in G.players:
        rlk_input[player] = {}
        for node in G.partition[player]:
            rlk_input[player][node.name] = {'M': M, 'Mprime': Mprime,
                                            'Level': Level, 'L0Dist': L0Dist,
                                            'SDist': SDist}
    return rlk_input


def _rlk_parallel(il):
    newgame = rlk(il[0], il[1], il[2], parallel=True)
    newgame.train_node(il[3], il[4])
    return newgame.G.node_dict[il[3]].LevelCPT


def rlk_parallel(G, ps, N, level_stop, level_start=1):
    """ Solves RLK in parallel.  Returns a Game where each node has
    attribute LevelCPT with entries from level_start to level_stop

    :arg G:  A semi-NFG
    :type G: semiNFG
    :arg player_specs: dictionary of dictionaries containing
    the level, satisficing distribution, level-0 strategy,
    and degrees of rationality of each player.
    :type player_specs: dict
    :arg level_stop: How high to train all player.
    :type level_stop: int
    :level_start: The starting level.  If level_start>1, the nodes in G
    must already have an attribute LevelCPT with a key
    'Level' + str(level_start-1)
    :type level_start: int

   For details on ps, see pynfg.levelksolutions.rlk

   """

    G1 = copy.deepcopy(G)
    from multiprocessing import Pool
    import pynfg
    dnode_list = [node.name for node in G1.nodes
                  if type(node) == pynfg.classes.decisionnode.DecisionNode]
    for lvl in np.arange(level_start, level_stop + 1):
        endlist = []
        inputlist = [G1, ps, N]
        p=Pool()
        for nd in dnode_list:
            holder = copy.deepcopy(inputlist)
            holder.extend([nd, lvl])
            endlist.append(holder)
        CPTs = p.map(_rlk_parallel, endlist)
        cpt_idx = 0
        for node in dnode_list:
            try:
                G1.node_dict[node].LevelCPT
            except AttributeError:
                G1.node_dict[node].LevelCPT = {}
            G1.node_dict[node].LevelCPT = CPTs[cpt_idx]
            cpt_idx += 1

    return G1
