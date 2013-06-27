# -*- coding: utf-8 -*-
"""
Implements the SemiNFG class

Part of: PyNFG - a Python package for modeling and solving Network Form Games

Created on Wed Nov 21 09:49:05 2012

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""

from __future__ import division
import numpy as np
import scipy as sp
import networkx as nx
import inspect
import matplotlib.pyplot as plt
from pynfg import DecisionNode, DeterNode, ChanceNode

class SemiNFG(object):
    """Implements the semi-NFG formalism created by D. Wolpert
    
    For an example, see PyNFG/bin/stackelberg.py     
    
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

    def _set_node_dict(self):
        """Set the node dictionary: keys are node names, values are nodes.
        """
        self.node_dict = {}
        for n in self.nodes:
            self.node_dict[n.name] = n

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

    def utility(self, player, nodeinput=None):
        """Evaluate the utility of the specified player

        :arg player: The name of a player with a utility function specified.
        :type player: str.
        :arg nodeinput: Optional. Keys are node names. Values are node values. 
           The values in nodeinput merely override the current node values, so 
           nodeinput does not need to specify values for every argument to a 
           player's utility function.
        :type nodeinput: dict
        
        """
        if nodeinput is None:
            nodeinput = {}
        if not self.u_functions:
            raise AssertionError('This is a semi-Bayes net, not a semi-NFG')
        kw = {}
        nodenames = inspect.getargspec(self.u_functions[player])[0]
        for nam in nodenames:        
            if nam in nodeinput:
                kw[nam] = nodeinput[nam]
            else:
                kw[nam] = self.node_dict[nam].get_value()
        u = self.u_functions[player](**kw)
        return u
            
    def children(self, nodename):
        """Retrieve the set of children of a given node.
        
        :arg nodename: the parent node for which children are desired.
        :type nodename: str
        :returns: a set of nodes that are the children of the input node in 
           the SemiNFG object.
        
        This is equivalent to calling ``SemiNFG.edges[nodename]``
        
        """
        kids = self.edges[nodename]
        return kids
        
    def parents(self, nodename):
        """Retrieve the set of parents of a given node.
        
        :arg nodename: the name of the child node for which parents are desired.
        :type nodename: str
        :returns: a set of nodes that are the parents of the input node in 
           the SemiNFG object.
        
        This is equivalent to calling ``set(node.parents.values())``
        
        """
        parents = set(self.node_dict[nodename].parents.values())
        return parents
    
    def descendants(self, nodename):
        """Retrieve the set of descendants of a given node.
        
        :arg nodename: the name of the ancestor node for which descendants are 
           desired.
        :type nodename: str
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
        
        for kid in self.edges[nodename]:
            future, visit_dict = kid_visit(kid, future, visit_dict)
        return future
        
    def ancestors(self, nodename):
        """Retrieve the set of ancestors of a given node.
        
        :arg nodename: the name of the descendent node for which ancestors are 
           desired.
        :type nodename: str
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
                
        for par in self.node_dict[nodename].parents.values():
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
                adict[name] = self.node_dict[name].get_value()
            return adict
            
    def get_decisionCPTs(self, mode=None):
        """Retrieve the CPTs for decision nodes
        
        :arg mode: optional string. If None, then keys are names of decision 
           nodes, and values are CPTs of corresponding decision nodes. If 
           'basename' then the output will be a dictionary with basename keys. 
           Values will be CPTs of the first instance (game chronology/topology) 
           of the corresponding basename.
        :type mode: str
        :returns: dict with nodenames or basenames as keys and current CPTs of
           corresponding decision nodes as output.
        
        """
        cptdict = {}
        if mode == 'basename':
            try:
                for bn in self.bn_part.keys():
                    if self.bn_part[bn][0].player != 'nature':
                        cptdict[bn] = self.bn_part[bn][0].CPT
            except AttributeError:
                raise TypeError('Use mode="basename" for iterSemiNFG only')
        else:
            for p in self.players:
                for n in self.partition[p]:
                    cptdict[n.name] = n.CPT
        return cptdict
        
    def set_CPTs(self, cptdict):
        """Set CPTs for nodes in the SemiNFG by node name
        
        :arg cptdict: dictionary with node names as keys and CPTs as values
        :type cptdict: dict
        """
        for name in cptdict.keys():
            self.node_dict[name].CPT = cptdict[name]
        
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
    
    def loglike(self, nodeinput=None):
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
        if nodeinput is None:
            nodeinput = {}
        problist = []
        for n in self.iterator:
            if n.name in nodeinput:
                problist.append(n.logprob(valueinput=nodeinput[n.name]))
            else:
                problist.append(n.logprob())
        r = np.sum(problist)
        return r
        
    def sample(self, start=None, nodenames=None, exclude=None):
        """Sample the net to obtain a draw from the joint distribution.
        
        :arg start: (Optional) if unspecified, the entire net will be sampled 
           from the prior. Otherwise, start is a list of nodenodes that serve as 
           the starting points for the sampling. That is, the sampling will 
           commence from the nodes in start and continue until all of the 
           descendants of the nodes in start have been sampled exactly once.
        :type start: list or set
        :arg nodenames: a list of node names for which output is desired. By
           default, no output is provided.
        :type nodenames: list or set
        :arg exclude: a list of node names that are to be held constant at their
           current values, i.e. excluded from the sampling.
        :type exclude: list or set
        :returns: a dict keyed by node names in nodenames input. Values are 
           values of nodes in node names.
        
        .. warning::
           
           The decision nodes must have CPTs before using this function.
        
        """
        if not exclude:
            exclude = []
        if not start:
            for n in self.iterator:
                if n.name not in exclude:
                    n.draw_value()
        else:
            children = set()
            starters = set([self.node_dict[nam] for nam in start])
            for nam in start:
                children.update(self.descendants(nam))
            for n in self.iterator:
                if n in children.union(starters):
                    if n.name not in exclude:
                        n.draw_value()
        if nodenames:
            outdict = dict(zip(nodenames, [self.node_dict[x].get_value() for \
                                            x in nodenames]))
            return outdict
        else:
            outdict = self.get_values()
        
            
    def draw_graph(self, subgraph=None):
        """Draw the DAG representing the topology of the SemiNFG.

        :arg subgraph: (Optional) node names of subset of nodes to include in 
           drawing. Default is self.nodes. If not specified, all nodes are 
           graphed. 
        :type subgraph: set or list
        
        .. note::
            
           This method uses the :py:mod:`matplotlib.pyplot` and 
           :py:mod:`networkx` packages.
           
        """
        G = nx.DiGraph()
        if not subgraph:
            nodelist = self.node_dict.values()
        else:
            nodelist = []
            for name in subgraph:
                nodelist.append(self.node_dict[name])
        nodeset = set(nodelist)
        for n in nodeset:
            for child in nodeset.intersection(self.edges[n.name]):
                G.add_edge(n.name,child.name)
        pos = nx.spring_layout(G, iterations=100)
 #       nx.draw_networkx(G, pos)
        fig = nx.draw_graphviz(G, prog='dot')
        plt.show()
        return fig