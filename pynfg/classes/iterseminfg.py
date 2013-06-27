# -*- coding: utf-8 -*-
"""
Implements the iterSemiNFG class

Part of: PyNFG - a Python package for modeling and solving Network Form Games

Created on Mon Feb 18 10:37:29 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""

from __future__ import division
import numpy as np
import scipy as sp
from seminfg import *

class iterSemiNFG(SemiNFG):
    """Implements the iterated semi-NFG formalism created by D. Wolpert
    
    For an example, see PyNFG/bin/hideandseek.py     
    
    :arg nodes: members are :class:`nodes.ChanceNode`, 
       :class:`nodes.DecisionNode`, or :class:`nodes.DeterNode`. The basename 
       and time attributes must be set for all nodes used in an iterSemiNFG.
    :type nodes: set
    :arg r_functions: One entry for each player. Keys are player names. 
       Values are keyword functions from the basename variables to real 
       numbers. 
    :type r_functions: dict
    
    An iterated semi-NFG is a semi-NFG created from iteratively gluing a kernel 
    to a base. It is a Markov iterated semi-NFG if the t'th copy of the kernel 
    is conditionally independent of all nodes in the t-2'th copy and earlier. 
    Instead of utility functions, iterated semi-NFGs use reward functions.
       
    .. note::
    
       This class is a subclass of :py:class:`seminfg.SemiNFG`. It inherits all 
       of the SemiNFG functionality except :py:meth:`seminfg.SemiNFG.utiity()` 
       is replaced by :py:meth:`seminfg.iterSemiNFG.reward()`.
    
    .. note::
    
       An object that consists of all of these elements except the reward
       functions is called a iterated semi-Bayes net.  When initialized with 
       `r_functions=None`, the result is an iterated semi-Bayes net.
       
    .. note::
       
       For a node in nodes, the parent attribute, e.g. 
       :py:attr:`nodes.ChanceNode.parents`, must not have parents that are not 
       in the set of nodes passed to :class:`seminfg.SemiNFG`.
       
    Some useful methods:
    
    * :py:meth:`seminfg.SemiNFG.ancestors()`
    * :py:meth:`seminfg.SemiNFG.descendants()`
    * :py:meth:`seminfg.SemiNFG.children()`
    * :py:meth:`seminfg.SemiNFG.loglike()`
    * :py:meth:`seminfg.SemiNFG.sample()`
    * :py:meth:`seminfg.SemiNFG.draw_graph()`
    * :py:meth:`seminfg.iterSemiNFG.reward()`
    * :py:meth:`seminfg.iterSemiNFG.sample_timesteps()`
       
    Upon initialization, the following private methods are called:
    
    * :py:meth:`seminfg.SemiNFG._set_node_dict()`
    * :py:meth:`seminfg.SemiNFG._set_partition()`
    * :py:meth:`seminfg.SemiNFG._set_edges()`
    * :py:meth:`seminfg.SemiNFG._topological_sort()`
    * :py:meth:`seminfg.iterSemiNFG._set_time_partition()`
    * :py:meth:`seminfg.iterSemiNFG.self._set_bn_part()`
        
    """
    def __init__(self, nodes, r_functions=None):
        self.nodes = nodes
        self.starttime = min([x.time for x in self.nodes])
        self.endtime = max([x.time for x in self.nodes])
        self._set_node_dict()
        self._set_edges()
        self._topological_sort()
        self._set_partition()
        self.players = [p for p in self.partition.keys() if p!='nature']
        self._set_time_partition()
        self._set_bn_part()
        self.r_functions = r_functions

    def _set_time_partition(self):
        """Set the time_partition :py:attr:`seminfg.iterSemiNFG.time_partition`
        
        :py:attr:`seminfg.iterSemiNFG.time_partition` is a partition of the 
        nodes into their corresponding timesteps. It is a dictionary, where 
        keys are integers 0 and greater corresponding to timesteps, and values 
        are lists of nodes that belong in that timestep, where the order of the
        list is given by the topological order in 
        :py:attr:`seminfg.iterSemiNFG.iterator`
        
        """
        self.time_partition = {}
        for n in self.iterator:
            if n.time not in self.time_partition.keys():
                self.time_partition[n.time] = [n]
            else:
                self.time_partition[n.time].append(n)
                
    def _set_bn_part(self):
        """Set the bn_part :py:attr:`seminfg.iterSemiNFG.bn_part`
        
        :py:attr:`seminfg.iterSemiNFG.bn_part` is a partition of the 
        nodes into groups according to nodes in a theoretical base/kernel. It 
        is a dictionary, where keys are basenames, and values are lists of 
        nodes that correspond to that basename. The order of the list is given
        by the time attribute.
        
        """
        self.bn_part = {}
        for n in self.nodes:
            if n.basename not in self.bn_part.keys():
                self.bn_part[n.basename] = [n]
            else:
                self.bn_part[n.basename].append(n)
        for bn in self.bn_part.keys():
            self.bn_part[bn].sort(key=lambda nod: nod.time)

    def reward(self, player, t, nodeinput=None):
        """Evaluate the reward of the specified player in the specified time.

        :arg player: The name of a player with a reward function specified.
        :type player: str.
        :arg nodeinput: Optional. Keys are basenames. Values are node values. 
           The values in nodeinput merely override the current node values, so 
           nodeinput does not need to specify values for every argument to a 
           player's reward function.
        :type nodeinput: dict
        
        """
        if nodeinput is None:
            nodeinput = {}
        if not self.r_functions:
            raise AssertionError('This is a semi-Bayes net, not a semi-NFG')
        kw = {}
        nodenames = inspect.getargspec(self.r_functions[player])[0]
        for nam in nodenames:        
            if nam in nodeinput:
                kw[nam] = nodeinput[nam]
            else:
                kw[nam] = self.bn_part[nam][t-self.starttime].get_value()
        r = self.r_functions[player](**kw)
        return r
    
    def npv_reward(self, player, start, delta, nodeinput=None):
        """Return the npv of rewards from start using delta discount factor
        
        :arg player: the name of the player to evaluate
        :type player: str
        :arg start: the starting time step
        :type start: int
        :arg delta: the discount factor for the npv calculation
        :type delta: float
        :arg nodeinput: Optional dict of node name, node values for use in 
           calculating the rewards
        """
        if nodeinput is None:
            nodeinput = {}
        count = 0
        npvreward = 0
        for t in range(start, self.endtime+1):
            npvreward += (delta**count)*self.reward(player, t, nodeinput)
            count += 1
        return npvreward
            
    def sample_timesteps(self, start, stop=None, basenames=None):
        """Sample the nodes from a starting time through a stopping time.
        
        :arg start: the first timestep to be sampled  
        :type start: integer
        :arg stop: (Optional) the last timestep to be sampled. If unspecified, 
           the net will be sampled to completion.
        :type stop: integer
        :arg basenames: (Optional) a list of strings that give the basenames 
           the user wants to collect as output. If omitted, there is no output.
        :returns: a dict keyed by base names in basenames input. Values are 
           time series of values from start to stop of nodes that share that 
           basename.
        
        .. warning::
           
           The decision nodes must have CPTs before using this function.
        
        """
        if stop==None or stop>self.endtime:
            stop = self.endtime
        if basenames:
#            import pdb; pdb.set_trace()
            outdict = dict(zip(basenames, [[] for x in range(len(basenames))]))
            for t in range(start, stop+1):
                for n in self.time_partition[t]:
                    if n.basename in basenames:
                        outdict[n.basename].append(n.draw_value())
                    else:
                        n.draw_value()
            return outdict
        else:
            for t in range(start, stop+1):
                for n in self.time_partition[t]:
                    n.draw_value()

    def get_values(self, nodenames=None):
        """Retrieve the values of the nodes comprising the SemiNFG.
        
        :arg nodenames: (Optional) The names of the nodes whose values should 
           be returned. If no names are specified, all node values are returned.
        :type nodenames: set or list
        :returns: dict where keys are node names and values are node values
        
        """
        if not nodenames:
            return dict(map(lambda x: (x.name, x.get_value()), self.nodes))
        else:
            adict = {}
            for name in nodenames:
                if name in self.node_dict.keys():
                    adict[name] = self.node_dict[name].get_value()
                elif name in self.bn_part.keys():
                    adict[name] = [x.get_value() for x in self.bn_part[name]]
                else:
                    print '%s is neither nodename nor basename' %name
            return adict
            
    def set_CPTs(self, cptdict):
        """Set CPTs for nodes in the iterSemiNFG by node name or basename
        
        :arg cptdict: dictionary with node names or basenames as keys and CPTs 
           as values. When a basename is given, the corresponding CPT is used 
           for all of the corresponding nodes.
        :type cptdict: dict
        """
        for name in cptdict.keys():
            if name in self.node_dict.keys():
                self.node_dict[name].CPT = cptdict[name]
            elif name in self.bn_part.keys():
                for t in range(len(self.bn_part[name])):
                    self.bn_part[name][t].CPT = cptdict[name]
            else:
                print ('%s is neither a node name nor a basename.' %name)