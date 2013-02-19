# -*- coding: utf-8 -*-
"""
Implements the ChanceNode class

Part of: PyNFG - a Python package for modeling and solving Network Form Games

Created on Mon Feb 18 10:35:19 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""

from __future__ import division
import inspect
import numpy as np
import scipy as sp
import scipy.stats.distributions as randvars
from node import *

class ChanceNode(Node):
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
            self._check_disc_parents()
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
            valslist = self.dict2list_vals(parentinput)
            indo = self.get_CPTindex(valslist, onlyparents=True)
            cdf = np.cumsum(self.CPT[indo])
            cutoff = np.random.rand()
            idx = np.nonzero( cdf >= cutoff )[0][0]
            r = self.space[idx]
        if setvalue:
            self.value = r
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
            valslist = self.dict2list_vals(parentinput, valueinput)
            indo = self.get_CPTindex(valslist)
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
        elif type(newvalue==self.space[0]) is bool:
            if newvalue in self.space:
                self.value = newvalue
            else:
                errorstring = "the new value is not in "+self.name+"'s space"
                raise ValueError(errorstring)
        elif any((newvalue==y).all() for y in self.space):
            self.value = newvalue
        else:
            raise ValueError("the new value is not in "+self.name+"'s space")