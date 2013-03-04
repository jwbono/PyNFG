# -*- coding: utf-8 -*-
"""
A Stackelberg-style example of a standard SemiNFG

Created on Mon Feb 25 17:58:20 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""
from __future__ import division

from pynfg import DecisionNode, ChanceNode, DeterNode
from pynfg import SemiNFG, iterSemiNFG
import matplotlib.pyplot as plt
import time

actions = np.arange(6)
markets = [(20,2), (10,1), (5,1/2)] #demand (intercept,slope)
c1 = 2 #marginal costs
c2 = 2 

MCPT = np.ones(len(markets))/len(markets)
Mparents = []
Mspace = markets
Mip = (MCPT, Mparents, Mspace)
M = ChanceNode('M', CPTip=Mip, description='determine market conditions')

Q1 = DecisionNode('Q1', '1', actions, [M], 'Stackelberg leader q')
Q2 = DecisionNode('Q2', '2', actions, [Q1], 'Stackelberg follower q')

def demand(q1, q2, m): #inverse demand
    Q = q1+q2
    a = m[0]
    b = m[1]
    price = a-b*Q
    return price
    
Dfunc = demand
Dparams =  {'q1': Q1, 'q2': Q2, 'm': M}
Dcont = True    
D = DeterNode('D', Dfunc, Dparams, Dcont, description='inv market demand')

def util1(Q1,D):
    profit = Q1*D-c1*Q1
    return profit

def util2(Q2,D):
    profit = Q2*D-c2*Q2
    return profit
    
u_funcs = {'1': util1, '2': util2}

nodeset = set([M,Q1,Q2,D])

G = SemiNFG(nodeset, u_funcs)

G.draw_graph()

#G.node_dict['Q1'].randomCPT(mixed=False)
#G.node_dict['Q2'].randomCPT(mixed=False)
#G.node_dict['Q2'].perturbCPT(0.5, mixed=False)
#
#G.sample()
#
#def welfare(G):
#    G.sample()
#    w = G.utility('1')+G.utility('2')
#    return w
#    
#def dens(i):
#    return np.power(i,2)
#
#S = 20000
#X = 10
#M = 40
#burn = 1000
#
#tipoff = time.time()
#intelMC, funcoutMC = iq_MC(G, S, X, M, integrand=welfare)
#halftime = time.time()
#print halftime-tipoff
#intelMH, funcoutMH = iq_MH(G, S, X, M, 0.2, dens, integrand=welfare)
#buzzer = time.time()
#print 'MH as percent of total time: ',(buzzer-halftime)/(buzzer-tipoff)
#
#weightsMC = dens(intelMC['Q1'])
#weightsMH = dens(intelMH['Q1'][burn::])
#
#plt.figure()
#plt.hist(intelMC['Q1'], normed=True, weights=weightsMC)
#plt.hist(intelMH['Q1'][burn::], normed=True, weights=weightsMH)
#
#plt.figure()
#plt.hist(funcoutMC.values(), normed=True, weights=weightsMC)
#plt.hist(funcoutMH.values()[burn::], normed=True, weights=weightsMH)