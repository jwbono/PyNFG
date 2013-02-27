# -*- coding: utf-8 -*-
"""
A Stackelberg-style example of a standard SemiNFG

Created on Mon Feb 25 17:58:20 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""
from __future__ import division

from classes import *
from rlsolutions import *
from pgtsolutions.intelligence import *
import matplotlib.pyplot as plt
import time

actions = np.arange(6)
markets = [(20,2), (10,1), (5,1/2)] #demand (intercept,slope)
c1 = 2 #marginal costs
c2 = 2 

MCPT = np.array([1/3, 1/3, 1/3])
Mparents = []
Mspace = markets
Mip = (MCPT, Mparents, Mspace)
M = ChanceNode('M', CPTip=Mip, description='determine market conditions')

Q1 = DecisionNode('Q1', '1', actions, [M], 'Stackelberg leader choose q')
Q2 = DecisionNode('Q2', '2', actions, [Q1], 'Stackelberg follower choose q')

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

#G.draw_graph()

G.node_dict['Q1'].randomCPT(mixed=False)
G.node_dict['Q2'].randomCPT(mixed=False)
G.node_dict['Q2'].perturbCPT(0.5, mixed=False)

G.sample()
print 

def welfare(G):
    w = G.utility('1')+G.utility('2')
    return w

#go = time.time()
#intel, funcout = iq_MC(G, 10000, 10, 20, integrand=welfare)
#print time.time()-go
#
#plt.hist(intel['Q1'], normed=True)
#plt.hist(intel['Q2'], normed=True)
#plt.hist(funcout.values(), normed=True)