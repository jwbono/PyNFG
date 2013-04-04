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
           object. If no valueinput is specified, then the output does not 
           include a value for the node itself.
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
                valuelist.append(par.get_value()) 
        if valueinput is not None:
            valuelist.append(valueinput)
        return valuelist
    
    def get_CPTindex(self, values, onlyparents=False):
        """Get the CPT index that corresponds to the (parent, node) values
        
        :arg values: a list whose members are values for the parents of the 
           decision node and the decision node itself, in the order given by 
           the `:py:attr:DecisionNode.parents` OrderedDict
        :type values: list
        :arg onlyparents: set to true if only parent indices are desired, i.e.
           the returned index slices into the last dimension of the CPT
        :type onlyparents: bool
           
        """
        if self.continuous:
            raise AttributeError('cont. nodes do not have CPTs')
        ind = []
        i = 0
        for par in self.parents.values():
            ind.append(par.valueindex)
#            if isinstance(par.space[0]==values[i], bool):
#                ind.append(par.space.index(values[i]))
#            else:
#                truth = [(x==values[i]).all() for x in par.space]
#                ind.append(truth.index(True))
#            i += 1
#        if not onlyparents:
#            if isinstance(self.space[0]==values[-1], bool):
#                ind.append(self.space.index(values[-1]))
#            else:
#                truth = [(x==values[-1]).all() for x in self.space]
#                ind.append(truth.index(True))
        indo = tuple(ind)
        return indo
        
    def set_value(self, value):
        """Set the current value of the Node object
        
        :arg value: a legitimate value of the Node object, i.e. the 
           value must be in :py:attr:`classes.Node.space`.
        
        .. warning::
            
           When arbitrarily setting values, some children may have zero 
           probability given their parents. This means the logprob may be -inf. 
           If using, :py:meth:`seminfg.SemiNFG.loglike()`, this results in a 
           divide by zero error.
        
        """
        if not self.continuous:
            self.set_valueindex(value)
        self.value = value                
        
    def set_valueindex(self, value):
        """Set the valueindex attribute of the discrete Node object
        
        :arg value: a legitimate value of the Node object, i.e. the value must 
           be in :py:attr:`classes.Node.space`. Otherwise an error occurs
        
        """
        if isinstance(value==self.space[0], bool):
            try: 
                self.valueindex = self.space.index(value)
            except ValueError:
                print 'the value %s is not in the space of %s' \
                        %(str(value),self.name)
        else:
            truth = [(x==value).all() for x in self.space]
            try: 
                self.valueindex = truth.index(True)
            except ValueError:
                print 'the value %s is not in the space of %s' \
                        %(str(value),self.name)
    
    def get_value(self):
        """Get the current value of the Node object
        
        """
        if self.continuous:
            return self.value
        else:
            return self.space[self.valueindex]
        
    def get_valueindex(self, value=None):
        """Get the valueindex attribute of the discrete Node object
        
        :arg value: a legitimate value of the Node object, i.e. the value must 
           be in :py:attr:`classes.Node.space`. Otherwise an error occurs. If 
           no value is provided, the current valueindex is returned.
        
        """
        if value:
            if isinstance(value==self.space[0], bool):
                try: 
                    ind = self.space.index(value)
                except ValueError:
                    print 'the value %s is not in the space of %s' \
                            %(str(value),self.name)
            else:
                truth = [(x==value).all() for x in self.space]
                try: 
                    ind = truth.index(True)
                except ValueError:
                    print 'the value %s is not in the space of %s' \
                            %(str(value),self.name)
            return ind
        else:
            return self.valueindex