# -*- coding: utf-8 -*-
"""
Implements the DeterNode class

Part of: PyNFG - a Python package for modeling and solving Network Form Games

Created on Mon Feb 18 10:34:13 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""

import numpy as np
from node import *

class DeterNode(Node):
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
          :py:meth:`classes.DecisionNode.prob()` or
          :py:meth:`classes.ChanceNode.prob()`.
       
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
        C1 = ChanceNode('C1', distip=distip1, 
                        description='CN given by norm loc=0.5, scale=1')
       
        def func2(var1=1, var2=0):
            r = np.sign(var1+var2)
            return r
            
        params2 = {'var1': C1}
        continuous2 = False
        space2 = [-1, 0, 1]
        F2 = DeterNode('F2', func2, params2, continuous2, space=space2, 
                       description='a disc. DeterNode child of C1')
    
    Upon initialization, the following private method is called: 
    :py:meth:`classes.DeterNode._set_parent_dict()`
    
    Some useful methods are:
       
    * :py:meth:`classes.DeterNode.draw_value()` 
    * :py:meth:`classes.DeterNode.prob()`
    * :py:meth:`classes.DeterNode.logprob()`
        
    """
    def __init__(self, name, func, params, continuous, space=None, \
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
        if space is None:
            space = []
        if isinstance(space, list):
            self.space = space
        else:
            raise TypeError('The space must be a list')
        self.parents = self._set_parent_dict(params.values())
        self.continuous = continuous
#        self.value = None
#        self.draw_value()        
        self.description = description
        self.time = time
        self.basename = basename
        
    def __str__(self):
        return self.name
    
    def draw_value(self, parentinput=None, setvalue=True):
        """Draw a value from the :class:`classes.DeterNode` object
        
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
           :py:attr:`classes.DeterNode.value`. True by default.
        :type setvalue: bool
        :returns: the value of the deterministic node that corresponds to the 
           parent values.
        
        .. note::
        
           If parent values are specified in parentinput, those values must be 
           legitimate values of the parent. For discrete parents, the values 
           must correspond to an item in the parent's space attribute. 
        
        """
        if parentinput is None:
            parentinput = {}
        funinput = {}
        for par in self.params:
            if isinstance(self.params[par],Node):
                if par in parentinput:
                    funinput[par] = parentinput[par]
                else:
                    funinput[par] = self.params[par].get_value()
            else:
                funinput[par] = self.params[par]
        r = self.dfunction(**funinput)
        if setvalue:
            self.set_value(r)
            return self.value
        else:
            return r
        
    def prob(self, parentinput=None, valueinput=None):
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
        if parentinput is None:
            parentinput = {}
        funinput = {}
        for par in self.params:
            if isinstance(self.params[par], Node):
                if par in parentinput:
                    funinput[par] = parentinput[par]
                else:
                    funinput[par] = self.params[par].get_value()
            else:
                funinput[par] = self.params[par]
        if valueinput is None:
            valueinput = self.get_value()
        try:
            r = 1*(self.dfunction(**funinput) == valueinput).all()
        except AttributeError:
            r = 1*(self.dfunction(**funinput) == valueinput)
        return r
        
    def logprob(self, parentinput=None, valueinput=None):
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
        if parentinput is None:
            parentinput = {}
        r = self.prob(parentinput, valueinput)
        return np.log(r)