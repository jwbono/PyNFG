# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 09:49:05 2012

Implements SemiNFG and iterSemiNFG classes

Part of: PyNFG - a Python package for modeling and solving Network Form Games
Copyright (C) 2013 James Bono

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""
from __future__ import division
import numpy as np
import scipy as sp
import networkx as nx
import inspect
import matplotlib.pyplot as plt
import pylab as P 
from nodes import DecisionNode, ChanceNode, DeterNode

class SemiNFG(object):
    """Implements the semi-NFG formalism created by D. Wolpert
    
    :arg nodes: members are :class:`nodes.ChanceNode`, 
       :class:`nodes.DecisionNode`, or :class:`nodes.DeterNode`.
    :type nodes: set
    :arg u_functions: One entry for each player. Keys are player names. 
       Values are functions from **to be continued**
    :type u_functions: dict
    
    Formally, a semi-NFG consists of the following elements:
        
       * nodes - either :class:`nodes.ChanceNode` or 
         :class:`nodes.DecisionNode`. For convenience, there is also a 
         :class:`nodes.DeterNode` class to implement deterministic nodes.
       
       * edges - given by :py:attr:`SemiNFG.edges`.
       
       * for each node, a conditional probability distribution from the values
         of its parents - given by :py:meth:`nodes.DecisionNode.prob()` or
         :py:meth:`nodes.ChanceNode.prob()`.
       
       * a partition splitting the set of nodes into a set of *nature* nodes 
         and a set of nodes for each player in the game, given by
         :py:attr:`seminfg.SemiNFG.partition`.
       
       * utility functions, one for each player in the game, given by
         :py:attr:`seminfg.SemiNFG.u_functions`.
       
    .. note::
       An object that consists of all of these elements except the utility 
       functions is called a semi-Bayes net. When initialized with 
       `u_functions=None`, the result is a semi-Bayes net.
           
    .. note::
       
       For a node in nodes, the parent attribute, e.g. 
       :py:attr:`nodes.ChanceNode.parents`, must not have parents that are 
       not in the set of nodes passed to :class:`seminfg.SemiNFG`.
       
    Example::
        
        from nodes import *
        import scipy.stats.distributions as randvars
        from types import IntType
        
        dist1 = randvars.randint
        params1 = [0, 1]
        distip1 = (dist1, params1)
        C1 = ChanceNode('C1', distip=distip1, description='root CN randint from 5 to 10')
        
        D1 = DecisionNode('D1', '1', [0, 1], parents=[C1], description='child node of C1. belongs to p1')
                            
        D2 = DecisionNode('D2', '2', [0, 1], parents=[C1], description='child node of C1. belongs to p2')
        
        def funcf(var1, var2, var3):
            if (var1+var2+var3)%2 == 0:
                return 'even'
            else:
                return 'odd'                    
        
        paramsf = {'var1': D1, 'var2': D2, 'var3': C1}
        continuousf = False
        spacef = ['even', 'odd']
        F1 = DeterNode('F1', funcf, paramsf, continuousf, space=spacef, description='a disc. DeterNode child of D1, D2, C1')
                       
        def ufunc1(F1):
            if F1 is 'even':
                return 1
            else:
                return -1
                
        def ufunc2(F1):
            if F1 is 'odd':
                return 1
            else:
                reutrn -1
        
        u_funcs = {'1': ufunc1, '2': ufunc2}
        
        nodeset = set([C1, D1, D2, F1])
        
        G = SemiNFG(nodeset, ufuncs)
       
    Some useful methods:
    
    * :py:meth:`seminfg.SemiNFG.ancestors()`
    * :py:meth:`seminfg.SemiNFG.descendants()`
    * :py:meth:`seminfg.SemiNFG.children()`
    * :py:meth:`seminfg.SemiNFG.loglike()`
    * :py:meth:`seminfg.SemiNFG.sample()`
    * :py:meth:`seminfg.SemiNFG.draw_graph()`
       
    Upon initialization, the following private methods are called:
    
    * :py:meth:`seminfg.SemiNFG._set_node_dict()`
    * :py:meth:`seminfg.SemiNFG._set_partition()`
    * :py:meth:`seminfg.SemiNFG._set_edges()`
    * :py:meth:`seminfg.SemiNFG._topological_sort()`
        
    """
    def __init__(self, nodes, u_functions=None):
        self.nodes = nodes
        self._set_node_dict()
        self._set_partition()
        self.players = [p for p in self.partition.keys() if p!='nature']
        self.u_functions = u_functions
        self._set_edges()
        self._topological_sort()
#        self._check_nodeparents

    def utility(self, player, nodeinput={}):
        """Evaluate the utility of the specified player

        :arg player: The name of a player with a utility function specified.
        :type player: str.
        :arg nodeinput: Optional. Keys are node names. Values are node values. 
           The values in nodeinput merely override the current node values, so 
           nodeinput does not need to specify values for every argument to a 
           player's utility function.
        :type nodeinput: dict
        
        """
        if not self.u_functions:
            raise AssertionError('This is a semi-Bayes net, not a semi-NFG')
        kw = {}
        nodenames = inspect.getargspec(self.u_functions[player])[0]
        for nam in nodenames:        
            if nam in nodeinput:
                kw[nam] = nodeinput[nam]
            else:
                kw[nam] = self.node_dict[nam].value
        u = self.u_functions[player](**kw)
        return u

    def _set_node_dict(self):
        """Set the node dictionary: keys are node names, values are nodes.
        """
        self.node_dict = {}
        for n in self.nodes:
            self.node_dict[n.name] = n
            
    def children(self, node):
        """Retrieve the set of children of a given node.
        
        :arg node: the parent node for which children are desired.
        :type node: :class:`nodes.ChanceNode`, :class:`nodes.DecisionNode`, or
           :class:`nodes.DeterNode`
        :returns: a set of nodes that are the children of the input node in 
           the SemiNFG object.
        
        This is equivalent to calling ``SemiNFG.edges[node.name]``
        
        """
        kids = self.edges[node.name]
        return kids
        
    def parents(self, node):
        """Retrieve the set of parents of a given node.
        
        :arg node: the child node for which parents are desired.
        :type node: :class:`nodes.ChanceNode`, :class:`nodes.DecisionNode`, or
           :class:`nodes.DeterNode`
        :returns: a set of nodes that are the parents of the input node in 
           the SemiNFG object.
        
        This is equivalent to calling ``set(node.parents)``
        
        """
        parents = set(node.parents)
        return parents
    
    def descendants(self, node):
        """Retrieve the set of descendants of a given node.
        
        :arg node: the ancestor node for which descendants are desired.
        :type node: :class:`nodes.ChanceNode`, :class:`nodes.DecisionNode`, or
           :class:`nodes.DeterNode`
        :returns: a set of nodes that are the descendants of the input node in 
           the SemiNFG object.
        
        """
        visit_dict = dict(map(lambda x: (x.name, False), self.nodes))
        future = set()
        
        def kid_visit(n, future, visit_dict):
            """Recursively retrieve the children, children of children, etc.
            
            :arg n: the parent node for which children are desired.
            :type n: :class:`nodes.ChanceNode`, :class:`nodes.DecisionNode`, 
               or :class:`nodes.DeterNode`
            :arg future: the set of descendents, growing in recursion.
            :type future: set
            :arg visit_dict: keys are node names, value is True if visited
            :type visit_dict: dict            
            :returns: updated versions of future and visit_dict
            """
            if not visit_dict[n.name]:
                visit_dict[n.name] = True
                for m in self.edges[n.name]:
                    future, visit_dict = kid_visit(m, future, visit_dict)
                future.add(n)
            return future, visit_dict
        
        for kid in self.edges[node.name]:
            future, visit_dict = kid_visit(kid, future, visit_dict)
        return future
        
    def ancestors(self, node):
        """Retrieve the set of ancestors of a given node.
        
        :arg node: the descendent node for which ancestors are desired.
        :type node: :class:`nodes.ChanceNode`, :class:`nodes.DecisionNode`, or
           :class:`nodes.DeterNode`
        :returns: a set of nodes that are the ancestors of the input node in 
           the SemiNFG object.
        
        """
        visit_dict = dict(map(lambda x: (x.name, False), self.nodes))
        past = set()

        def par_visit(n, past, visit_dict):
            """Recursively retrieve the parents, parents of parents, etc.
            
            :arg n: the child node for which parents are desired.
            :type n: :class:`nodes.ChanceNode`, :class:`nodes.DecisionNode`, 
               or :class:`nodes.DeterNode`
            :arg past: the set of ancestors, growing in recursion.
            :type past: set
            :arg visit_dict: keys are node names, value is True if visited
            :type visit_dict: dict            
            :returns: updated versions of past and visit_dict
            """
            if not visit_dict[n.name]:
                visit_dict[n.name] = True
                for m in n.parents.values():
                    past, visit_dict = par_visit(m, past, visit_dict)
                past.add(n)
            return past, visit_dict
                
        for par in node.parents.values():
            past, visit_dict = par_visit(par, past, visit_dict)
        return past

    def get_leaves(self):
        """Retrieve the leaves of the SemiNFG.
        
        :returns: set of leaf nodes, which are :class:`nodes.ChanceNode`, 
           :class:`nodes.DecisionNode`, or :class:`nodes.DeterNode`
        
        """
        leaves = set()
        for n in self.edges:
            if not self.edges[n]:
                leaves.add(self.node_dict[n])
        return leaves        
    
    def get_roots(self):
        """Retrieve the roots of the SemiNFG.
        
        :returns: set of root nodes, which are :class:`nodes.ChanceNode`, 
           :class:`nodes.DecisionNode`, or :class:`nodes.DeterNode`
        
        """
        roots = set()
        for n in self.nodes:
            if not n.parents:
                roots.add(n)
        return roots
        
    def get_values(self):
        """Retrieve the values of the nodes comprising the SemiNFG.
        
        :returns: dict where keys are node names and values are node values
        
        """
        return dict(map(lambda x: (x.name, x.value), self.nodes))
        
    def set_values(self, value_dict):
        """Set the values of a subset of the nodes comprising the SemiNFG.
        
        :arg value_dict: keys are node names, and values are node values.
        :type value_dict: dict
        
        .. note::
            
           For discrete valued nodes, the value must be in the node's space, or
           else a ValueError will be raised by that node's set_value method, 
           e.g. :py:meth:`nodes.DecisionNode.set_value()`.
           
        .. warning::
            
           When arbitrarily setting values, some children may have zero 
           probability given their parents, and the log likelihood may be -inf,
           which results in a divide by zero error for the method 
           :py:meth:`seminfg.SemiNFG.loglike()`.
        
        """
        for n in value_dict:
            self.node_dict[n].set_value(value_dict[n])
    
    def loglike(self, nodeinput={}):
        """Compute the log likelihood of the net using the values in nodeinput.
        
        :arg nodeinput: Keys are node names. Values are node values. This 
           optional argument can be any subset of nodes in the SemiNFG. 
           For nodes not specified, the current value is used.
        :type nodeinput: dict
        :returns: the log likelihood of the values given the net.
        
        .. note::
            
           For discrete valued nodes, the value must be in the node's space, or
           else an error will be be raised by that node's prob() method, 
           e.g. :py:meth:`nodes.DecisionNode.prob()`.
           
        .. warning::
            
           When arbitrarily setting values, some children may have zero 
           probability given their parents, and the log likelihood may be -inf,
           which results in a divide by zero error.
           
        .. warning::
           
           The decision nodes must have CPTs before using this function.
           
        """
        problist = []
        for n in self.nodes:
                problist.append(n.logprob(nodeinput))
        r = np.sum(problist)
        return r
        
    def sample(self, start=None):
        """Sample the net to obtain a draw from the joint distribution.
        
        :arg start: (Optional) if unspecified, the entire net will be sampled 
           from the prior. Otherwise, start is a list of nodes that serve as 
           the starting points for the sampling. That is, the sampling will 
           commence from the nodes in start and continue until all of the 
           descendants of the nodes in start have been sampled exactly once.
        :type start: list
        :returns: a list of values drawn from the joint distribution given by 
           the net. The order of the values in the list is given by the order 
           of the nodes in the attribute list 
           :py:attr:`seminfg.SemiNFG.iterator`.
        
        .. warning::
           
           The decision nodes must have CPTs before using this function.
        
        """
        if not start:
            values = []
            for n in self.iterator:
                values.append(n.draw_value())
            return values
        else:
            children = set()
            values = []
            for n in start:
                chidren = children.update(self.descendants(n))
            for n in self.iterator:
                if n in children.union(set(start)):
                    values.append(n.draw_value())
                else: 
                    values.append(n.value)
            return values
            
    def draw_graph(self, subgraph=None):
        """Draw the DAG representing the topology of the SemiNFG.

        :arg subgraph: (Optional) a set or list of nodes to be graphed. If not
           specified, all nodes are graphed.
        :type subgraph: set or list
        
        .. note::
            
           This method uses the :py:mod:`matplotlib.pyplot` and 
           :py:mod:`networkx` packages.
           
        """
        G = nx.DiGraph()
        if not subgraph:
            subgraph = self.nodes
        for n in subgraph:
            for child in subgraph.intersection(self.edges[n.name]):
                G.add_edge(n.name,child.name)
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos)
        plt.show()
        
    def _set_partition(self):
        """Set the partition attribute :py:attr:`seminfg.SemiNFG.partition`
        
        The attribute :py:attr:`seminfg.SemiNFG.partition` is a dictionary
        with keys as the names of players and *nature*, and values are sets of
        nodes that belong to those players. 
        
        """
        self.partition = {}
        for n in self.nodes:
            if n.player not in self.partition.keys():
                self.partition[n.player] = set([n])
            else:
                self.partition[n.player].add(n)
        
    def _set_edges(self):
        """Set the edges attribute :py:attr:`seminfg.SemiNFG.edges`
        
        The attribute :py:attr:`seminfg.SemiNFG.edges` is a dictionary
        with keys as the names of nodes, and values are the set of child nodes.
        
        """
        self.edges = dict(map(lambda x: (x.name, set()), self.nodes))
        for n in self.nodes:
            for par in n.parents.values():
                self.edges[par.name].add(n)
    
    def _topological_sort(self):
        """Set the edges attribute :py:attr:`seminfg.SemiNFG.iterator`
        
        The attribute :py:attr:`seminfg.SemiNFG.iterator` is a list of the 
        nodes in topological order, i.e. if a node has parents, then those 
        parents are earlier in the list than the node itself. This list is used 
        to simultate the net.
        
        """
        self.iterator = []
        S = self.get_leaves()
        visit_dict = dict(map(lambda x: (x.name, False), self.nodes))
        
        def top_visit(n, top_order, visit_dict):
            if not visit_dict[n.name]:
                visit_dict[n.name] = True
                for m in n.parents.values():
                    top_order, visit_dict = top_visit(m, top_order, visit_dict)
                top_order.append(n)
            return top_order, visit_dict
        
        for n in S:
            self.iterator, visit_dict = top_visit(n, self.iterator, visit_dict)
            

class iterSemiNFG(SemiNFG):
    """Implements the iterated semi-NFG formalism created by D. Wolpert
    
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
       
    Example::
        
        import scipy.stats.distributions as randvars
        
        dist1 = randvars.randint
        params1 = [0, 4]
        space1 = range(10)
        distip1 = (dist1, params1, space1)
        C1 = ChanceNode('C10', distip=distip1, description='root CN randint from 0 to 3', basename='C1', time=0)
        
        D1 = DecisionNode('D10', '1', [0, 1], parents=[C1], description='child node of C1. belongs to p1', basename='D1', time=0)
                            
        D2 = DecisionNode('D20', '2', [0, 1], parents=[C1], description='child node of C1. belongs to p2', basename='D2', time=0)
        
        def funcf(var1, var2, var3):
            total = var1+var2+var3
            if total>10:
                return total-8
            elif total<1:
                return total+2
            else:
                return total                    
        
        paramsf = {'var1': D1, 'var2': D2, 'var3': C1}
        continuousf = False
        spacef = range(11)
        F1 = DeterNode('F10', funcf, paramsf, continuousf, space=spacef, description='a disc. DeterNode child of D1, D2, C1', basename='F1', time=0)
        
        nodeset = set([C1,D1,D2,F1])
                       
        for t in range(1,4):
            
            params1 = [0, F1]
            distip1 = (dist1, params1, space1)
            C1 = ChanceNode('C1%s' %t, distip=distip1, description='CN randint from 0 to 3', basename='C1', time=t)
            nodeset.add(C1)
                            
            D1 = DecisionNode('D1%s' %t, '1', [0, 1], parents=[C1], description='child node of C1. belongs to p1', basename='D1', time=t)     
            nodeset.add(D1)
                            
            D2 = DecisionNode('D2%s' %t, '2', [0, 1], parents=[C1], description='child node of C1. belongs to p2', basename='D2', time=t) 
            nodeset.add(D2)
        
            D2.randomCPT(setCPT=True)
            D2.draw_value()
            D1.randomCPT(setCPT=True)
            D1.draw_value()
            
            paramsf = {'var1': D1, 'var2': D2, 'var3': C1}
            F1 = DeterNode('F1%s' %t, funcf, paramsf, continuousf, space=spacef, description='a disc. DeterNode child of D1, D2, C1', basename='F1', time=t)
            nodeset.add(F1)
            
        def reward1(F1=0):
            if F1<3:
                if F1%2 == 0:
                    x=1
                else: 
                    x=-1
            elif F1<=7:
                if F1%2 == 1:
                    x=1
                else:
                    x=-1
            else:
                if F1%2 == 0:
                    x=1
                else:
                    x=-1
            return x
            
        def reward2(F1=0):
            return -1*reward1(F1)
        
        rfuncs = {'1': reward1, '2': reward2}
        G = iterSemiNFG(nodeset, rfuncs)
        G.reward('1', 2)
       
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
    * :py:meth:`seminfg.iterSemiNFG.self._set_basename_partition()`
        
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
        self._set_basename_partition()
        self.r_functions = r_functions

    def reward(self, player, time, nodeinput={}):
        """Evaluate the reward of the specified player in the specified time.

        :arg player: The name of a player with a reward function specified.
        :type player: str.
        :arg nodeinput: Optional. Keys are node names. Values are node values. 
           The values in nodeinput merely override the current node values, so 
           nodeinput does not need to specify values for every argument to a 
           player's reward function.
        :type nodeinput: dict
        
        """
        if not self.r_functions:
            raise AssertionError('This is a semi-Bayes net, not a semi-NFG')
        kw = {}
        nodenames = inspect.getargspec(self.r_functions[player])[0]
        for nam in nodenames:        
            if nam in nodeinput:
                kw[nam] = nodeinput[nam]
            else:
                kw[nam] = self.basename_partition[nam][time].value
        r = self.r_functions[player](**kw)
        return r
    
    def npv_reward(self, player, start, delta, nodeinput={}):
        count = 0
        npvreward = 0
        for tin in range(start, self.endtime+1):
            count += 1
            npvreward += (delta**count)*self.reward(player, tin)
        return npvreward
        
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
        for n in self.nodes:
            if n.time not in self.time_partition.keys():
                self.time_partition[n.time] = [n]
            else:
                self.time_partition[n.time].append(n)
        for t in range(self.endtime):
            self.time_partition[t] = \
                [n for n in self.iterator if n in self.time_partition[t]]
                
    def _set_basename_partition(self):
        """Set the basename_partition :py:attr:`seminfg.iterSemiNFG.basename_partition`
        
        :py:attr:`seminfg.iterSemiNFG.basename_partition` is a partition of the 
        nodes into groups according to nodes in a theoretical base/kernel. It 
        is a dictionary, where keys are basenames, and values are lists of 
        nodes that correspond to that basename. The order of the list is given
        by the time attribute.
        
        """
        self.basename_partition = {}
        for n in self.nodes:
            if n.basename not in self.basename_partition.keys():
                self.basename_partition[n.basename] = [n]
            else:
                self.basename_partition[n.basename].append(n)
        for bn in self.basename_partition.keys():
            self.basename_partition[bn].sort(key=lambda nod: nod.time)
            
    def sample_timesteps(self, start, stop=None, basenames=None):
        """Sample the nodes from a starting time through a stopping time.
        
        :arg start: the first timestep to be sampled  
        :type start: integer
        :arg stop: (Optional) the last timestep to be sampled. If unspecified, 
           the net will be sampled to completion.
        :type stop: integer
        :arg basenames: (Optional) a list of strings that give the basenames 
           the user wants to collect as output. If omitted, there is no output.
        :returns: a list of lists of values drawn from the joint distribution 
           given by the net. The order of the lists in the return list is given 
           by the time step, and the order of the values in the list is given 
           by the order of the nodes in the attribute list 
           :py:attr:`seminfg.iterSemiNFG.iterator`.
        
        .. warning::
           
           The decision nodes must have CPTs before using this function.
        
        """
        values = []
        if stop==None or stop>self.endtime:
            stop = self.endtime
        if basenames:
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