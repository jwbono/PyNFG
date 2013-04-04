# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:42:34 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""

from collections import OrderedDict

class Node(object):
    """Implements a generic node of the semi-NFG formalism created by D. Wolpert
    
    .. note::
        
       This is the superclass. Nodes are generally instantiated in one of the 
       subclasses, ChanceNode, DecisionNode or DeterNode.
    
    :arg name: the name of the Node, usually descriptive, e.g. C5 for
       the fifth chance node (C for chance), or C21 for the second chance node 
       in the first time step, etc.
    :type name: str
    :arg parents: a list of the Node's parents
    :type parents: list
    :arg continuous: True if Node takes on continuous values. False if
       discrete.
    :type continuous: bool 
    
    Upon initialization, the following private method is called: 
    :py:meth:`nodes.DeterNode._set_parent_dict()`
    
    Some useful methods are:
       
    * :py:meth:`classes.Nodes.dict2list_vals()` 
    * :py:meth:`classes.Nodes.get_CPTindex()`
        
    """
    def __init__(self, name, parents, continuous):
        self.name = name
        self.parents = self._set_parent_dict(parents)
        self.continuous = continuous
    
    def _set_parent_dict(self, parents):
        """Set the parent OrderedDict based on the params entered by user
        
        :arg parents: list of parameter values for the Node. For ChanceNode and
           DecisionNode objects, this is just a list of parent nodes. For 
           DeterNode objects this list may also contain non-node objects, such 
           as fixed parameters of the DeterNode function.
        :type parents: list
        :returns: OrderedDict. Keys are node names. Values are node objects.
        
        """
        r = OrderedDict()
        for par in parents:
            if isinstance(par, Node):
                r[par.name] = par
        return r
            
    def _check_disc_parents(self):
        """Check that parents entered by user are discrete
        
        :arg parent: a dictionary with keys as parent names and values as parent 
           nodes
        :type parent: dict
        
        """
        for par in self.parents.values():
            if par.continuous is True:
                raise RuntimeError("The parent named %s is continuous!" %par.name)
                
    def dict2list_vals(self, parentinput=None, valueinput=None):
        """Convert parent/value dict entered by user to a list of values
        
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
        if parentinput is None:
            parentinput = {}
        valuelist = []
        for par in self.parents.values():
            if par.name in parentinput:
                valuelist.append(parentinput[par.name]) 
            else:            
                valuelist.append(par.value) 
        if valueinput is not None:
            valuelist.append(valueinput)
        return valuelist
    
    def get_CPTindex(self, values, onlyparents=False):
        """Get the CPT index that corresponds to the (parent, node) values
        
        :arg values: the parents OrderedDict attribute for the node, e.g.
           `:py:attr:DecisionNode.parents` or `:py:attr:ChanceNode.parents`
        :type values: dict
        :arg onlyparents: set to true if 
        :type onlyparents: bool
        :arg values: a list whose members are values for the parents of the 
           decision node and the decision node itself, in the order given by 
           the `:py:attr:DecisionNode.parents` OrderedDict
           
        """
        if self.continuous:
            raise AttributeError('cont. nodes do not have CPTs')
        ind = []
        i = 0
        for par in self.parents.values():
            if type(par.space[0]==values[i]) is bool:
                ind.append(par.space.index(values[i]))
            else:
                truth = [(x==values[i]).all() for x in par.space]
                ind.append(truth.index(True))
            i += 1
        if not onlyparents:
            if type(self.space[0]==values[-1]) is bool:
                ind.append(self.space.index(values[-1]))
            else:
                truth = [(x==values[-1]).all() for x in self.space]
                ind.append(truth.index(True))
        indo = tuple(ind)
        return indo