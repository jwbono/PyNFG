"""
Implements relaxed level k for a semi-network form game

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""

class rlk(object):
    """ Finds the relaxed level-k solution for a semi-NFG.
    References
    ----------
    Lee and Wolpert, "Game theoretic modeling of pilot behavior
    during mid-air encounters," Decision-Making with Imperfect
    Decision Makers, T. Guy, M. Karny and D.H.Wolpert (Ed.’s),
    Springer (2011).

    : arg G:  A semi-NFG
    :type G: semiNFG
    :arg player_specs: dictionary of dictionaries containing
    The level, satisficing distribution, level-0 strategy,
    and degrees of rationality of each player.
    See below for details.
    :type player_specs: dict

    player_spec is a dictionary of dictionaries.  The keys are
    the players.  For each player the dictionary has five entries:

    Level : int
        The level of the player
    M : int
        The number of times to sample the satisficing distribution
    M' : int
        The number of times to sample the net for each satisficing
        distribution.
    L0Dist : ndarray
        If 'uni' then the level 0 distribution is set to a uniform
        distribution. If ndarray, then the level 0 CPT is set to
        L0Dist. If no distribution is specified,
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
        pass
