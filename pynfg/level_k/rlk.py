import copy as copy
import warnings


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

    player_spec is a dictionary of dictionaries.  The keys are
    the players.  For each player the dictionary has five entries:

    Level : int
        The level of the player
    M : int
        The number of times to sample the satisficing distribution
    Mprime : int
        The number of times to sample the net for each satisficing
        distribution.
    L0Dist : ndarray, None
        If ndarray, then the level 0 CPT is set to
        L0Dist. If L0Dist is None,
        then the level 0 CPT is set to the uniform distribution.
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
    def __init__(self, G, player_specs):
        self.player_specs = player_specs
        self.G = copy.copy(G)
        self._set_new_attributes
        self._set_L0_CPT

    def _set_new_attributes(self):
        """ Sets the level and rationality of a decision node

        It assigns an attribute "Level", "M" and "Mprime" to each decision
        node.
        """
        G = self.G
        ps = self.player_specs
        for player in self.player_specs:
            G.node_dict[list(G.partition[player])[0]].Level = ps[player]['Level']
            G.node_dict[list(G.partition[player])[0]].M = ps[player]['M']
            G.node_dict[list(G.partition[player])[0]].Mprime = ps[player]['Mprime']

    def _set_L0_CPT(self):
        G = self.G
        ps = self.player_specs
        for player in ps:
            try:
                if ps[player]['L0Dist'] is None:
                    list(G.partition[player])[0].uniformCPT()
                else:
                    list(G.partition[player])[0].CPT = ps[player]['L0Dist']
            except KeyError:
                warnings.warn("No entry for L0Dist for player %s,\
                    setting to uniform" % player)
                list(G.partition[player])[0].uniformCPT()

    def _set_satisficing_func(self):
        G = self.G
        ps = self.player_specs
        for player in ps:
            if hasattr(ps[player]['SDist'], '__call__'):
                list(G.partition[player])[0].SDist = ps[player]['SDist']

