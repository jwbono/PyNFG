# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:39:14 2012
Copyright (C) 2013 James Bono
GNU Affero General Public License

Part of: PyNFG - a Python package for modeling and solving Network Form Games
Implements ChanceNode, DecisionNode and DeterNode classes

"""
__author__="""James Bono (jwbono@gmail.com)"""

from __future__ import division
import numpy as np
import scipy as sp
import scipy.stats.distributions as randvars
from collections import OrderedDict
import inspect

class ChanceNode(object):
    """Implements a chance node of the semi-NFG formalism created by D. Wolpert

    The :class:`nodes.ChanceNode` can be initialized with either a 
    conditional probability distribution (CPT) or a distribution object 
    from :py:mod:`scipy.stats.distributions` (discrete and continuous types 
    are both supported).
    
    :arg name: the name of the ChanceNode, usually descriptive, e.g. C5 for
       the fifth chance node, or C21 for the second chance node in the first 
       time step, etc.
    :type name: str
    :arg CPTip: The input parameters for a chance node based on a CPT. It 
       is a tuple with the following three elements:
       * CPTip[0]: np.array giving CPT - in order given by parent spaces
       * CPTip[1]: list of parents - in order of dims of CPT
       * CPTip[2]: list of space - in order of last dim of CPT
    :type CPTip: tuple
    :arg distip: The input parameters for a chance node based on a 
       distribution from the :py:mod:`scipy.stats.distributions` module. It is
       a tuple with the following three elements::
           
          * distip[0]: :py:mod:`scipy.stats.distributions` distribution object
          * distip[1]: list of distribution params - order given by 
             distribution args.
          * distip[2]: (None if distribution object is continuous) list of space 
             if distribution object is discrete.
          
    :type distip: tuple
    :arg description: a description of the chance node, usually including a
       summary description of the distribution, space, parents and children.
    :type description: str
    :arg time: the timestep to which the node belongs. This is generally 
       only used when the node is in a :class:`seminfg.iterSemiNFG`.
    :type time: int
    :arg basename: This is only used when the node is in a 
       :class:`seminfg.iterSemiNFG`. It references a theoretical node in the 
       base or kernel.
    :type basename: str
    
    .. note::
       
       For a :class:`nodes.ChanceNode` based on a CPT, the parents 
       must be discrete valued nodes. The dimensions of the CPT must 
       correspond to the order of the parents. The order of the CPT in each 
       dimension must correspond to the order of the parent space for that 
       dimension.
       
    Formally, a chance node has the following properties:
    
    * belongs to the *nature* player
    * has a space of possible values
    * has a conditional probability distribution from the values of its 
      parents - given by :py:meth:`nodes.DecisionNode.prob()` or
      :py:meth:`nodes.ChanceNode.prob()`.
    
    Example::
        
        import scipy.stats.distributions as randvars
                        
        D1 = DecisionNode('D1', '1', [-1, 0, 1], parents=[], description='This is a child node of C1')
        
        dist1 = randvars.norm
        params1 = [D1, 2]
        distip1 = (dist2, params2)
        C1 = ChanceNode('C1', distip=distip1, description='CN norm rv with scale=2 and loc=D1')
                        
    or::
        
        import scipy.stats.distributions as randvars
        
        dist1 = randvars.randint
        params1 = [5, 10]
        distip1 = (dist1, params1)
        C1 = ChanceNode('C1', distip=distip1, description='root CN randint from 5 to 10')
        
        dist2 = randvars.hypergeom
        params2 = [C1, 3, 3]
        space2 = [0, 1, 2, 3]
        distip2 = (dist2, params2, space2)
        C2 = ChanceNode('C2', distip=distip1, description='CN hypergeom M=C1, n=3, N=3')
                        
    Upon initialization, the following private method is called: 
    :py:meth:`nodes.ChanceNode._set_parent_dict()`
    
    Some useful methods are:
       
    * :py:meth:`nodes.ChanceNode.draw_value()` 
    * :py:meth:`nodes.ChanceNode.prob()`
    * :py:meth:`nodes.ChanceNode.logprob()` 
        
    """
    def __init__(self, name, CPTip=None, distip=None, \
                 description='no description', time=None, basename=None, \
                 verbose=False):
        if verbose:
            try:
                print 'Name: '+ name + '\nDescription: '+ description
            except TypeError:
                print('name and description should be strings')
        self.name = name
        self.description = description
        self.player = 'nature'
        self.time = time
        self.basename = basename
        if distip is None:
            self.CPT = CPTip[0]
            self.parents = self._set_parent_dict(CPTip[1])
            _check_parents(self.parents)
            self.space = CPTip[2]
            self.continuous = False
        else:
            self.CPT = None
            self.distribution = distip[0]
            self.params = distip[1]
            parlist = filter(lambda x: type(x) is DecisionNode \
                         or type(x) is ChanceNode \
                         or type(x) is DeterNode, \
                         self.params)
            self.parents = self._set_parent_dict(parlist)
            self.continuous = (randvars.rv_continuous in \
                                inspect.getmro(type(self.distribution)))
            if self.continuous is False:
                self.space = distip[2]
            else:
                self.space = []
        self.draw_value()
        
    def __str__(self):
        return self.name
        
    def draw_value(self, parentinput={}, setvalue=True):
        """Draw a value from the :class:`nodes.ChanceNode` object
        
        :arg parentinput: Optional. Specify values of the parents at which to 
           draw values from the conditional probability distribution. Keys are 
           parent names. Values are parent values. To specify values for only a 
           subset of the parents, only enter those parents in the dictionary. 
           If no parent values are specified, then the current values of the 
           parents are used. 
        :type parentinput: dict
        :arg setvalue: (Optional) determines if the random draw replaces
           :py:attr:`nodes.ChanceNode.value`. True by default.
        :type setvalue: bool
        :returns: an element of :py:attr:`nodes.ChanceNode.space` if the 
           ChanceNode object is discrete. For continuous ChanceNode objects, it 
           returns a value at which the pdf is nonzero. 
        
        .. note::
        
           If parent values are specified in parentinput, those values must be 
           legitimate values of the parent. For discrete parents, the values 
           must correspond to an item in the parent's space attribute. 
        
        """
        if self.CPT is None:
            if not parentinput:
                arglist = map(lambda x: x.value \
                              if type(x) in (DecisionNode, ChanceNode, DeterNode) \
                              else x, self.params)
            else:
                arglist = map(lambda x: parentinput[x.name] \
                              if type(x) in (DecisionNode, ChanceNode, DeterNode) \
                              else x, self.params)
            argtuple = tuple(arglist)
            r = self.distribution.rvs(*argtuple)
        else:
            valslist = dict2list_vals(self.parents, parentinput)
            indo = get_CPTindex(self, valslist, onlyparents=True)
#            ind = []
#            for par in self.parents.values():
#                if par.name in parentinput:
#                    truth = [(x==parentinput[par.name]).all() for x in par.space]
#                    ind.append(truth.index(True)) 
#                else:
#                    truth = [(x==par.value).all() for x in par.space]
#                    ind.append(truth.index(True))                                                 
#            indo = tuple(ind)
            cdf = np.cumsum(self.CPT[indo])
            cutoff = np.random.rand()
            idx = np.nonzero( cdf >= cutoff )[0][0]
            r = self.space[idx]
        if setvalue:
            self.set_value(r)
            return self.value
        else:
            return r
        
    def prob(self, parentinput={}, valueinput=None):
        """Compute the conditional probability of the current or specified value
        
        :arg parentinput: Optional. Specify values of the parents at which to 
           compute the conditional probability. Keys are parent names. Values 
           are parent values. To specify values for only a subset of the 
           parents, only enter those parents in the dictionary. If no parent 
           values are specified, then the current values of the parents are used. 
        :type parentinput: dict
        :arg valueinput: Optional. A legitimate value of the chance node 
           object. If no valueinput is specified, then the current value of the 
           node is used.
        :returns: the conditional probability of valueinput or the current
           value conditioned on parentinput or the current values of the parents.
        
        .. note::
        
           If parent values are specified in parentinput, those values must be 
           legitimate values of the parent. For discrete parents, the values 
           must correspond to an item in the parent's space attribute. 
        
        """
        if valueinput is None:
            valueinput = self.value
        if self.CPT is None:
            if not parentinput:
                arglist = map(lambda x: x.value \
                              if type(x) in (DecisionNode, ChanceNode, DeterNode) \
                              else x, self.params)
            else:
                arglist = map(lambda x: parentinput[x.name] \
                              if type(x) in (DecisionNode, ChanceNode, DeterNode) \
                              else x, self.params)
            args = tuple(arglist)
            if self.continuous:
                r = self.distribution.pdf(valueinput, *args)
            else:
                r = self.distribution.pmf(valueinput, *args)
        else:
            if valueinput is None:
                valueinput = self.value
            valslist = dict2list_vals(self.parents, parentinput, valueinput)
            indo = get_CPTindex(self, valslist)
#            ind = []
#            for par in self.parents.values():
#                if par.name in parentinput:
#                    truth = [(x==parentinput[par.name]).all() for x in par.space]
#                    ind.append(truth.index(True)) 
#                else:
#                    truth = [(x==par.value).all() for x in par.space]
#                    ind.append(truth.index(True)) 
#            truth = [(x==valueinput).all() for x in par.space]
#            ind.append(truth.index(True))
#            indo = tuple(ind)
            r = self.CPT[indo]
        return r
        
    def logprob(self, parentinput={}, valueinput=None):
        """Compute the conditional logprob of the current or specified value
        
        :arg parentinput: Optional. Specify values of the parents at which to 
           compute the conditional logprob. Keys are parent names. Values are 
           parent values. To specify values for only a subset of the parents, 
           only enter those parents in the dictionary. If no parent values are 
           specified, then the current values of the parents are used. 
        :type parentinput: dict
        :arg valueinput: Optional. A legitimate value of the chance node 
           object. If no valueinput is specified, then the current value of the 
           node is used.
        :returns: the log conditional probability of valueinput or the current
           value conditioned on parentinput or the current values of the parents.
        
        .. note::
        
           If parent values are specified in parentinput, those values must be 
           legitimate values of the parent. For discrete parents, the values 
           must correspond to an item in the parent's space attribute.
           
        This is equivalent to ``np.log(ChanceNode.prob())``
        
        """
        r = self.prob(parentinput, valueinput)
        return np.log(r)
        
    def set_value(self, newvalue):
        """Set the current value of the ChanceNode object
        
        :arg newvalue: a legitimate value of the ChanceNode object. If the 
           ChanceNode object is discrete, then newvalue must be in 
           :py:attr:`nodes.ChanceNode.space`. If the ChanceNode object is 
           continuous, no corrections are made for values at which the pdf is 0.
        
        .. warning::
            
           When arbitrarily setting values, some children may have zero 
           probability given their parents. This means the logprob may be -inf. 
           If using, :py:meth:`seminfg.SemiNFG.loglike()`, this results in a 
           divide by zero error.
        
        """
        if self.continuous:
            self.value = newvalue
        elif any((newvalue==y).all() for y in self.space):
            self.value = newvalue
        else:
            raise ValueError("the new value is not in "+self.name+"'s space")
        
    def _set_parent_dict(self, parents):
        """Set the parent OrderedDict based on the parents list entered by user
        
        :arg parents: list of parent nodes for the ChanceNodes
        :type parents: list
        :returns: OrderedDict in which the items follow the order of the parent 
           list entered by the user. Keys are node names. Values are node 
           objects.
        
        """
        r = OrderedDict()
        for par in parents:
            r[par.name] = par
        return r
            
class DeterNode(object):
    """Implements a deterministic node for the semi-NFG formalism created 
    by D. Wolpert
    
    :arg name: the name of the DeterNode, usually descriptive, e.g. F5 for
       the fifth deterministic node (F for fixed), or F21 for the second 
       deterministic node in the first time step, etc.
    :type name: str
    :arg func: a deterministic function - defined with defaults for each of 
       its inputs.
    :type func: function
    :arg params: keys are input keywords for func, values are parents or 
       fixed values if something other than defaults for non-parent inputs is
       desired.
    :type params: dict
    :arg continuous: True if function takes on continuous values. False if
       discrete.
    :type continuous: bool 
    :arg space: The list of elements in space if discrete. Empty list if 
       continuous 
    :type space: list
    :arg description: a description of the DeterNode, usually including a
       summary description of the function, space, parents and children.
    :type description: str.
    :arg time: the timestep to which the node belongs. This is generally 
       only used for :class:`seminfg.iterSemiNFG` objects.
    :type time: integer
    :arg basename: a reference to a theoretical node in the base or kernel.
    :type basename: str.
    
    Formally, a deterministic node has the following properties:
        
       * belongs to the *nature* player
       * has deterministic function from the values of its parents - given by 
          :py:meth:`nodes.DecisionNode.prob()` or
          :py:meth:`nodes.ChanceNode.prob()`.
       
    .. note::
        
       Deterministic nodes are not a part of the Semi-NFG formalism. Their 
       functionality is implicit in the relationship between stochastic parent
       nodes and their stochastic children, as is the convention in Bayesian 
       Networks. They can be seen as deterministically *transforming* the 
       values of parents into the parameters that are used by their children.
    
    Example::
            
        import scipy.stats.distributions as randvars
        
        dist1 = randvars.norm
        params1 = [0.5, 1]
        distip1 = (dist1, params1)
        C1 = ChanceNode('C1', distip=distip1, description='CN given by norm loc=0.5, scale=1')
       
        def func2(var1=1, var2=0):
            r = np.sign(var1+var2)
            return r
            
        params2 = {'var1': C1}
        continuous2 = False
        space2 = [-1, 0, 1]
        F2 = DeterNode('F2', func2, params2, continuous2, space=space2, description='a disc. DeterNode child of C1')
    
    Upon initialization, the following private method is called: 
    :py:meth:`nodes.DeterNode._set_parent_dict()`
    
    Some useful methods are:
       
    * :py:meth:`nodes.DeterNode.draw_value()` 
    * :py:meth:`nodes.DeterNode.prob()`
    * :py:meth:`nodes.DeterNode.logprob()`
        
    """
    def __init__(self, name, func, params, continuous, space=[], \
                 description='no description', time=None, basename=None, \
                 verbose=False):
        if verbose:
            try:
                print 'Name: '+ name + '\nDescription: '+ description
            except TypeError:
                print('name and description should be strings')
        self.name = name
        self.player = 'nature'
        self.dfunction = func
        self.params = params
        self.space = space
        self.parents = self._set_parent_dict(params.values())
        self.continuous = continuous
        self.draw_value()        
        self.description = description
        self.time = time
        self.basename = basename
        
    def __str__(self):
        return self.name
    
    def draw_value(self, parentinput={}, setvalue=True):
        """Draw a value from the :class:`nodes.DeterNode` object
        
        This function computes the value of the deterministic node given the
        current values of the parents or with the values provided in pareninput
        
        :arg parentinput: Optional. Specify values of the parents at which to 
           compute values from the function governing the DeterNode. Keys are 
           parent names. Values are parent values. To specify values for only a 
           subset of the parents, only enter those parents in the dictionary. 
           If no parent values are specified, then the current values of the 
           parents are used. 
        :type parentinput: dict
        :arg setvalue: (Optional) determines if the random draw replaces
           :py:attr:`nodes.DeterNode.value`. True by default.
        :type setvalue: bool
        :returns: the value of the deterministic node that corresponds to the 
           parent values.
        
        .. note::
        
           If parent values are specified in parentinput, those values must be 
           legitimate values of the parent. For discrete parents, the values 
           must correspond to an item in the parent's space attribute. 
        
        """
        funinput = {}
        for par in self.params:
            if type(self.params[par]) in (DecisionNode, DeterNode, ChanceNode):
                if par in parentinput:
                    funinput[par] = pareninput[par].value
                else:
                    funinput[par] = self.params[par].value
        r = self.dfunction(**funinput)
        if setvalue:
            self.set_value(r)
            return self.value
        else:
            return r
        
    def prob(self, parentinput={}, valueinput=None):
        """Compute the probability of the current or specified value
        
        Note that since this is a deterministic node, the probability is always
        either zero or one.
        
        :arg parentinput: Optional. Specify values of the parents at which to 
           compute the probability of the value of the DeterNode function. 
           Keys are parent names. Values are parent values. To specify values 
           for only a subset of the parents, only enter those parents in the 
           dictionary. If no parent values are specified, then the current 
           values of the parents are used. 
        :type parentinput: dict
        :arg valueinput: Optional. A legitimate value of the DeterNode 
           object. If no valueinput is specified, then the current value of the 
           node is used.
        :returns: the conditional probability of valueinput or the current
           value conditioned on parentinput or the current values of the 
           parents.
        
        .. note::
        
           If parent values are specified in parentinput, those values must be 
           legitimate values of the parent. For discrete parents, the values 
           must correspond to an item in the parent's space attribute. 
        
        """       
        funinput = {}
        for par in self.params:
            if type(self.params[par]) in (DecisionNode, DeterNode, ChanceNode):
                if self.params[par].name in parentinput:
                    funinput[par] = parentinput[self.params[par].name].value
                else:
                    funinput[par] = self.params[par].value
            else:
                funinput[par] = self.params[par]
        if valueinput is None:
            valueinput = self.value
        if self.dfunction(**funinput) == valueinput:
            r=1
        else:
            r=0
        return r
        
    def logprob(self, parentinput={}, valueinput=None):
        """Compute the conditional logprob of the current or specified value
        

        
        :arg parentinput: Optional. Specify values of the parents at which to 
           compute the logprob of the value of the DeterNode. Keys are parent 
           names. Values are parent values. To specify values for only a subset 
           of the parents, only enter those parents in the dictionary. If no 
           parent values are specified, then the current values of the parents 
           are used. 
        :type parentinput: dict
        :arg valueinput: Optional. A legitimate value of the DeterNode object. 
           If no valueinput is specified, then the current value of the node is 
           used.
        :returns: the log conditional probability of valueinput or the current
           value conditioned on parentinput or the current values of the 
           parents.
        
        .. note::
        
           If parent values are specified in parentinput, those values must be 
           legitimate values of the parent. For discrete parents, the values 
           must correspond to an item in the parent's space attribute.
           
        .. note::
            
           Since this is a deterministic node, the logprob is alwayseither zero 
           or -inf.
           
        This is equivalent to ``np.log(DeterNode.prob())``
        
        """
        r = self.prob(parentinput, valueinput)
        return np.log(r)
        
    def set_value(self, newvalue):
        """Set the current value of the DeterNode object
        
        :arg newvalue: a legitimate value of the DeterNode object. If the 
           DeterNode object is discrete, then newvalue must be in 
           :py:attr:`nodes.DeterNode.space`. If the DeterNode object is 
           continuous, no corrections are made for unattainable values.
        
        .. warning::
            
           When arbitrarily setting values, some children may have zero 
           probability given their parents. This means the logprob may be -inf. 
           If using, :py:meth:`seminfg.SemiNFG.loglike()`, this results in a 
           divide by zero error.
        
        """
        if self.continuous:
            self.value = newvalue
        elif any((newvalue==y).all() for y in self.space):
            self.value = newvalue
        else:
            raise ValueError("the new value is not in "+self.name+"'s space")
    
    def _set_parent_dict(self, parents):
        """Set the parent OrderedDict based on the params entered by user
        
        :arg parents: list of parameter values for the DeterNode function
        :type parents: list
        :returns: OrderedDict. Keys are node names. Values are node objects.
        
        """
        r = OrderedDict()
        for par in parents:
            if type(par) in (DecisionNode, DeterNode, ChanceNode):
                r[par.name] = par
        return r

class DecisionNode(object):
    """Implements a decision node of the semi-NFG formalism by D. Wolpert
    
    The :class:`nodes.DecisionNode` can be initialized with either a 
    conditional probability distribution (CPT) or a distribution object 
    from :py:mod:`scipy.stats.distributions` (discrete and continuous types 
    are both supported).
    
    :arg name: the name of the DecisionNode, usually descriptive, e.g. D5, 
       for player 5's decision node, or D51 for player 5's 1st decision node, 
       or D512 for player 5's 1st decision node in the 2nd time step, etc.
    :type name: str
    :arg player: the name of the player to which this DecisionNode belongs.
    :type player: str
    :arg space: the list of the possible values for the DecisionNode. The 
       order determines the order of the CPT when generated.
    :type space: list
    :arg parents: the list of the parents of the decision node. All entries 
       must be a :class:`nodes.DecisionNode` or a discrete 
       :class:`nodes.ChanceNode` or :class:`nodes.DeterNode`. The order of the 
       parents in the list determinesthe rder of the CPT when generated.
    :type parents: list
    :arg description: a description of the decision node, usually including 
       a summary description of the space, parents and children.
    :type description: str
    :arg time: the timestep to which the node belongs. This is generally 
       only used for :class:`seminfg.iterSemiNFG` objects.
    :type time: int
    :arg basename:  Reference to a theoretical node in the base or kernel.
    :type basename: str
    
    Formally, a decision node has the following properties:
        
       * belongs to a human player
       * has a space of possible values.
       * the conditional probability distribution from the values of its 
          parents - given by :py:meth:`nodes.DecisionNode.prob()` or
          :py:meth:`nodes.ChanceNode.prob()`, is not specified in the game. That
          distribution is given by the solution concept applied to the semi-NFG. 
          This lack of CPDs at decision nodes is the reason the semi-NFG is 
          said to be based on a semi-Bayes net.

    .. note::
           
       For a :class:`nodes.DecisionNode`, the parents nodes must be 
       discrete.
    
    Example::
        
        import scipy.stats.distributions as randvars
                        
        dist1 = randvars.randint
        params1 = [1, 4]
        space1 = [1, 2, 3]
        distip1 = (dist1, params1, space1)
        C1 = ChanceNode('C1', distip=distip1, description='root CN given by randint 1 to 4')
                        
        D1 = DecisionNode('D1', '1', [-1, 0, 1], parents=[C1], description='This is a child node of C1')
                            
    Upon initialization, the following private method is called: 
    :py:meth:`nodes.DecisionNode._set_parent_dict()`
    
    Some useful methods are:
       
    * :py:meth:`nodes.DecisionNode.draw_value()` 
    * :py:meth:`nodes.DecisionNode.prob()`
    * :py:meth:`nodes.DecisionNode.logprob()`
    * :py:meth:`nodes.DecisionNode.randomCPT()`
    * :py:meth:`nodes.DecisionNode.perturbCPT()`
        
    """
    def __init__(self, name, player, space, parents=[], \
                 description='no description', time=None, basename=None, \
                 verbose=False):
        if verbose:
            try:
                print 'Name: '+ name + '\nDescription: '+ description + \
                    '\nPlayer: '+player 
            except (AttributeError, TypeError):
                raise AssertionError('name, description, player should be strings')
        self.name = name
        self.description = description
        self.time = time
        self.basename = basename
        self.player = player
        self.space = space
        self.parents = self._set_parent_dict(parents)
        self._createCPT()
        _check_parents(self.parents)
        self.value = self.space[0]
        
    def __str__(self):
        return self.name
        
    def draw_value(self, parentinput={}, setvalue=True, mode=False):
        """Draw a value from the :class:`nodes.DecisionNode` object
        
        :arg parentinput: Optional. Specify values of the parents at which to 
           draw values using the CPT. Keys are parent names. Values are parent 
           values. To specify values for only a subset of the parents, only 
           enter those parents in the dictionary. If no parent values are 
           specified, then the current values of the parents are used. 
        :type parentinput: dict
        :arg setvalue: (Optional) determines if the random draw replaces
           :py:attr:`nodes.DecisionNode.value`. True by default.
        :type setvalue: bool
        :returns: an element of :py:attr:`nodes.DecisionNode.space`.
        
        .. note::
        
           The values specified in parentinput must correspond to an item in the 
           parent's space attribute.
           
        .. warning::
            
           The CPT is an np.zero array upon initialization. Therefore, one must 
           set the CPT wih :py:meth:`nodes.DecisionNode.randomCPT()` or 
           manually before calling this method.
        
        """
        if not self.CPT.any():
            raise RuntimeError('CPT for %s is just a zeros array' % self.name)
        ind = []
        valslist = dict2list_vals(self.parents, parentinput)
        indo = get_CPTindex(self, valslist, onlyparents=True)
#        for par in self.parents.values():
#            if par.name in parentinput:
#                ind.append(par.space.index(parentinput[par.name])) 
#            else:               
#                ind.append(par.space.index(par.value)) 
        indo = tuple(ind)
        if not mode:
            cdf = np.cumsum(self.CPT[indo])
            cutoff = np.random.rand()
            idx = np.nonzero( cdf >= cutoff )[0][0]
        else:
            idx = self.CPT[indo].argmax()
        r = self.space[idx]
        if setvalue:
            self.set_value(r)
            return self.value
        else:
            return r
        
    def randomCPT(self, mixed=False, setCPT=True):
        """Create a random CPT for the :class:`nodes.DecisionNode` object
        
        :arg mixed: Optional. Determines whether a mixed CPT, i.e. a CPT that 
           assigns nonzero weight to every value in 
           :py:attr:`nodes.DecisionNode.space`, or a pure CPT, i.e. a CPT that 
           assigns probability 1 to a single value in 
           :py:attr:`nodes.DecisionNode.space` for each of the parent values.
        :type mixed: bool
        :arg setCPT: Optional. Default is True. Determines whether the 
           :py:attr:`nodes.DecisionNode.CPT` attribut is set by the function
        :type setCPT: bool
        :returns: a mixed or pure CPT.
        
        """
        CPTshape = self.CPT.shape        
        shape_last = CPTshape[-1]
        other_dims = CPTshape[0:-1]
        z = np.zeros(CPTshape)
        if mixed is False:
            y = randvars.randint.rvs(0, shape_last, size=other_dims)
            if y.size > 1:
                z.reshape((-1, shape_last))[np.arange(y.size), y.flatten()]=1
            else:
                z.reshape((-1, shape_last))[0, y]=1
        else:
            M = 100000000
            x = randvars.randint.rvs(1, M, size=other_dims+(shape_last-1,))
            y = np.concatenate((np.zeros(other_dims+(1,)), x, \
                                M*np.ones(other_dims+(1,))), axis=-1)
            yy = np.sort(y, axis=-1)
            z = np.diff(yy, axis=-1)/M
        if setCPT:
            self.CPT = z
        else:
            return z
            
    def uniformCPT(self, setCPT=True):
        """Create a uniform CPT for the :class:`nodes.DecisionNode` object
        
        :arg setCPT: Optional. Default is True. Determines whether the 
           :py:attr:`nodes.DecisionNode.CPT` attribute is set by the function
        :type setCPT: bool
        :returns: a uniform mixed CPT.
        
        """
        z = np.zeros(self.CPT.shape)
        z += 1/(self.CPT.shape[-1])
        if setCPT:
            self.CPT = z
        else:
            return z
        
    def perturbCPT(self, noise, mixed=True, sliver=None):
        """Create a perturbation of the CPT attribute.
        
        :arg noise: The noise determines the mixture between the current CPT 
           and a random CPT, e.g. `new = self.CPT*(1-noise) + randCPT*noise`. 
           Noise must be a number between 0 and 1.
        :type noise: float
        :arg mixed: Optional. Determines if the perturbation is pure or mixed. 
           If pure, then the perturbed CPT is a pure CPT with some of the pure 
           weights shifted to other values. If mixed, then the perturbed CPT is 
           a mixed CPT with positive weight on all values. 
        :type mixed: bool
        :arg sliver: Optional. Determines the values of the parents for which 
           to perturb the current CPT. Keys are parent names. Values are parent 
           values. If empty, the entire CPT is perturbed. If sliver is nonempty, 
           but specifies values for only a subset of parents, the current values 
           are used for the remaining parents.
        :type sliver: dict
        
        .. warning::
            
           Functionality for pure perturbations is not yet implemented!
        
        """
#        if not mixed:
#            if not sliver:
#                shape_last = self.CPT.shape[-1]
#                other_dims = self.CPT.shape[0:-1]
#                y = randvars.randint.rvs(0, shape_last, size=other_dims)
#        else:
        randCPT = self.randomCPT(mixed=False)
        if not sliver:
            z = self.CPT*(1-noise) + randCPT*noise
        else:
            ind = []
            for par in self.parents:
                if par in sliver:
                    truth = [(x==sliver[par]).all() for x in \
                                                    self.parents[par].space]
                    ind.append(truth.index(True))
                else:
                    value = self.parents[par].value
                    truth = [(x==value).all() for x in \
                                                    self.parents[par].space]
                    ind.append(truth.index(True))
            indo = tuple(ind)
            z = self.CPT
            z[indo] = z[indo]*(1-noise) + randCPT[indo]*noise
        return z
        
    def prob(self, parentinput={}, valueinput=None):
        """Compute the conditional probability of the current or specified value
        
        :arg parentinput: Optional. Specify values of the parents at which to 
           compute the conditional probability. Keys are parent names. Values 
           are parent values. To specify values for only a subset of the 
           parents, only enter those parents in the dictionary. If only a 
           subset of parent values are specified, then the current values are 
           used for the remaining parents.
        :type parentinput: dict
        :arg valueinput: Optional. A legitimate value of the decision node 
           object. If no valueinput is specified, then the current value of the 
           node is used.
        :returns: the conditional probability of valueinput or the current
           value conditioned on parentinput or the current values of the 
           parents.
        
        .. note::
        
           If parent values are specified in parentinput, those values must 
           correspond to items in the space attributes of the parents.
           
        .. warning::
            
           The CPT is an np.zero array upon initialization. Therefore, one must 
           set the CPT wih :py:meth:`nodes.DecisionNode.randomCPT()` or 
           manually before calling this method.
        
        """        
        if not self.CPT.any():
            raise RuntimeError('CPT for %s is just a zeros array' % self.name)
        if valueinput is None:
            valueinput = self.value
        valslist = dict2list_vals(self.parents, parentinput, valueinput)
        indo = get_CPTindex(self, valslist)
        p = self.CPT[indo]
        return p  
        
    def logprob(self, parentinput={}, valueinput=None):
        """Compute the conditional logprob of the current or specified value
        
        :arg parentinput: Optional. Specify values of the parents at which to 
           compute the conditional logprob. Keys are parent names. Values are 
           parent values. To specify values for only a subset of the parents, 
           only enter those parents in the dictionary. If only a subset of 
           parent values are specified, then the current values are used for the 
           remaining parents.
        :type parentinput: dict
        :arg valueinput: Optional. A legitimate value of the decision node 
           object. If no valueinput is specified, then the current value of the 
           node is used.
        :returns: the conditional logprob of valueinput or the current
           value conditioned on parentinput or the current values of the 
           parents.
        
        .. note::
        
           If parent values are specified in parentinput, those values must 
           correspond to items in the space attributes of the parents.
           
        .. warning::
            
           The CPT is an np.zero array upon initialization. Therefore, one must 
           set the CPT wih :py:meth:`nodes.DecisionNode.randomCPT()` or 
           manually before calling this method.
        
        """        
        r = self.prob(parentinput, valueinput)
        return np.log(r)
        
    def set_value(self, newvalue):
        """Set the current value of the DecisionNode object
        
        :arg newvalue: a legitimate value of the DecisionNode object, i.e. the 
           value must be in :py:attr:`nodes.ChanceNode.space`.
        
        .. warning::
            
           When arbitrarily setting values, some children may have zero 
           probability given their parents. This means the logprob may be -inf. 
           If using, :py:meth:`seminfg.SemiNFG.loglike()`, this results in a 
           divide by zero error.
        
        """
        if any((newvalue==x).all() for x in self.space):
            self.value = newvalue
        else:
            raise ValueError("the new value is not in "+self.name+"'s space")  
        
    def _set_parent_dict(self, parents):
        """Set the parent OrderedDict based on the parents list entered by user
        
        :arg parents: list of parent nodes for the DecisionNode
        :type parents: list
        :returns: OrderedDict in which the items follow the order of the parent 
           list entered by the user. Keys are parent names. Values are parent 
           objects.
        
        """
        r = OrderedDict()
        if parents is not None:
            for par in parents:
                r[par.name] = par
        return r
        
    def _createCPT(self):
        """Create a CPT of the correct size with zeros for the DecisionNode
        
        Uses the order of the parents in the parent list as entered by the user 
        to initialize the DecisionNode object and the sizes of space attributes 
        of the parents to create a :py:func:`numpy.zeros()` array of the 
        appropriate size and shape.
        
        """
        CPT_size = []
        for par in self.parents:
            CPT_size.append(len(self.parents[par].space))
        CPT_size.append(len(self.space))
        self.CPT = np.zeros(CPT_size)

        
def _check_parents(parents):
    """Check that parents entered by user are discrete
    
    :arg parent: a dictionary with keys as parent names and values as parent 
       nodes
    :type parent: dict
    
    """
    for par in parents.values():
        if par in (ChanceNode, DeterNode) and par.continuous is True:
            raise RuntimeError("The parent named %s is continuous!" %par.name)
            
def dict2list_vals(parentdict, parentinput={}, valueinput=None):
    """Convert parent/value dict entered by user to a list of values
    
    :arg parentdict: the parents OrderedDict attribute for the node, e.g.
       `:py:attr:DecisionNode.parents` or `:py:attr:ChanceNode.parents`
    :type parentdict: dict
    :arg parentinput: Optional. Specify values of the parents. Keys are 
       parent names. Values are parent values. To specify values for only a 
       subset of the parents, only enter those parents in the dictionary. 
       If only a subset of parent values are specified, then the current 
       values are used for the remaining parents.
    :type parentinput: dict
    :arg valueinput: Optional. A legitimate value of the decision node 
       object. If no valueinput is specified, then the output does not include a 
       value for the node itself.
    :returns: a list of the values for the parents and the decision node 
       itself in the order determine by the :py:attr:DecisionNode.parents` 
       OrderedDict.
    
    """
    values = []
    for par in parentdict.values():
        if par.name in parentinput:
            values.append(parentinput[par.name]) 
        else:            
            values.append(par.value) 
    if valueinput is not None:
        values.append( valueinput )
    return values

def get_CPTindex(node, values, onlyparents=False):
    """Get the CPT index that corresponds to the (parent, node) values
    
    :arg parentdict: the parents OrderedDict attribute for the node, e.g.
       `:py:attr:DecisionNode.parents` or `:py:attr:ChanceNode.parents`
    :type parentdict: dict
    :arg values: a list whose members are values for the parents of the 
       decision node and the decision node itself, in the order given by 
       the `:py:attr:DecisionNode.parents` OrderedDict
       
    """
    ind = []
    i = 0
    for par in node.parents.values():
        truth = [(x==values[i]).all() for x in par.space]
        ind.append(truth.index(True))
        i += 1
    if not onlyparents:
        truth = [(x==values[-1]).all() for x in node.space]
        ind.append(truth.index(True))
    indo = tuple(ind)
    return indo
    
x = DecisionNode('test', '1', [0,1,2,3,4])