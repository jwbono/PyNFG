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

    The :class:`classes.ChanceNode` can be initialized with either a 
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
       
       For a :class:`classes.ChanceNode` based on a CPT, the parents 
       must be discrete valued nodes. The dimensions of the CPT must 
       correspond to the order of the parents. The order of the CPT in each 
       dimension must correspond to the order of the parent space for that 
       dimension.
       
    Formally, a chance node has the following properties:
    
    * belongs to the *nature* player
    * has a space of possible values
    * has a conditional probability distribution from the values of its 
      parents - given by :py:meth:`classes.DecisionNode.prob()` or
      :py:meth:`classes.ChanceNode.prob()`.
    
    Example::
        
        import pynfg
        import scipy.stats.distributions as randvars
                        
        D1 = DecisionNode('D1', '1', [-1, 0, 1], parents=[], 
                          description='This is a child node of C1')
        
        dist1 = randvars.norm
        params1 = [D1, 2]
        distip1 = (dist1, params1)
        C1 = ChanceNode('C1', distip=distip1, 
                        description='CN norm rv with scale=2 and loc=D1')
                        
    or::
        
        import pynfg
        import scipy.stats.distributions as randvars
        
        dist1 = randvars.randint
        params1 = [5, 10]
        distip1 = (dist1, params1)
        C1 = ChanceNode('C1', distip=distip1, 
                        description='root CN randint from 5 to 10')
        
        dist2 = randvars.hypergeom
        params2 = [C1, 3, 3]
        space2 = [0, 1, 2, 3]
        distip2 = (dist2, params2, space2)
        C2 = ChanceNode('C2', distip=distip1, 
                        description='CN hypergeom M=C1, n=3, N=3')
                        
    Upon initialization, the following private method is called: 
    :py:meth:`classes.ChanceNode._set_parent_dict()`
    
    Some useful methods are:
       
    * :py:meth:`classes.ChanceNode.draw_value()` 
    * :py:meth:`classes.ChanceNode.prob()`
    * :py:meth:`classes.ChanceNode.logprob()` 
        
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
            if isinstance(CPTip[2], list):
                self.space = CPTip[2]
            else:
                raise TypeError('The space, CPTip[2], must be a list')
            self.continuous = False
        else:
            self.CPT = None
            self.distribution = distip[0]
            self.params = distip[1]
            parlist = filter(lambda x: isinstance(x,Node), self.params)
            self.parents = self._set_parent_dict(parlist)
            self.continuous = (randvars.rv_continuous in \
                                inspect.getmro(type(self.distribution)))
            if self.continuous is False:
                self.space = distip[2]
            else:
                self.space = []
#        self.draw_value()
        
    def __str__(self):
        return self.name
        
    def draw_value(self, parentinput=None, setvalue=True):
        """Draw a value from the :class:`classes.ChanceNode` object
        
        :arg parentinput: Optional. Specify values of the parents at which to 
           draw values from the conditional probability distribution. Keys are 
           parent names. Values are parent values. To specify values for only a 
           subset of the parents, only enter those parents in the dictionary. 
           If no parent values are specified, then the current values of the 
           parents are used. 
        :type parentinput: dict
        :arg setvalue: (Optional) determines if the random draw replaces
           :py:attr:`classes.ChanceNode.value`. True by default.
        :type setvalue: bool
        :returns: an element of :py:attr:`classes.ChanceNode.space` if the 
           ChanceNode object is discrete. For continuous ChanceNode objects, it 
           returns a value at which the pdf is nonzero. 
        
        .. note::
        
           If parent values are specified in parentinput, those values must be 
           legitimate values of the parent. For discrete parents, the values 
           must correspond to an item in the parent's space attribute. 
        
        """
        if parentinput is None:
            parentinput = {}
        if self.CPT is None:
            arglist = []
            for val in self.params:
                if isinstance(val,Node):
                    if val.name in parentinput.keys():
                        arglist.append(parentinput[val.name])
                    else:
                        arglist.append(val.get_value())
                else:
                    arglist.append(val)
            argtuple = tuple(arglist)
            r = self.distribution.rvs(*argtuple)
            if setvalue:
                self.set_value(r)
                return self.get_value()
            else:
                return r
        else:
            indo = self.get_CPTindex(parentinput, valueinput=False)
            cdf = np.cumsum(self.CPT[indo])
            cutoff = np.random.rand()
            idx = np.nonzero( cdf >= cutoff )[0][0]
            if setvalue:
                self.set_valueindex(idx)
                return self.get_value()
            else:
                return self.space[idx]
                   
    def prob(self, parentinput=None, valueinput=None):
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
        if parentinput is None:
            parentinput = {}
        if self.CPT is None:
            if not parentinput:
                arglist = map(lambda x: x.get_value() \
                              if isinstance(x,Node) else x, self.params)
            else:
                arglist = map(lambda x: parentinput[x.name] \
                              if isinstance(x,Node) else x, self.params)
            args = tuple(arglist)
            if valueinput is None:
                valueinput = self.get_value()
            if self.continuous:
                r = self.distribution.pdf(valueinput, *args)
            else:
                r = self.distribution.pmf(valueinput, *args)
        else:
            indo = self.get_CPTindex(parentinput, valueinput)
            r = self.CPT[indo]
        return r
        
    def logprob(self, parentinput=None, valueinput=None):
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
        if parentinput is None:
            parentinput = {}
        r = self.prob(parentinput, valueinput)
        return np.log(r)
        