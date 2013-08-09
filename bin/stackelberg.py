# -*- coding: utf-8 -*-
"""
A Stackelberg-style example of a standard SemiNFG:
Market demand is determined by a root node. Firm 1 observes market demand and 
chooses a quantity, Q1. Firm 2 observes Q1 and chooses a quantity, Q2. Firm 2 
does not observe market demand. Q1 and Q2 are combined wih market demand to 
determine prices. Prices and quantities then determine profits.

Note: It is better to run this script line by line or customize your own run
script from the pieces contained herein rather than running the entire file. The
reason is that the PGT algorithms will take a long time with a large number of 
sample (S).

Created on Mon Feb 25 17:58:20 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""
from __future__ import division

import numpy as np
from pynfg import DecisionNode, ChanceNode, DeterNode
from pynfg import SemiNFG
import matplotlib.pyplot as plt
import time
import copy

###########################################
##PARAMETERS AND FUNCTIONS
###########################################
actions = range(6) #list of actions for each firm (quantities)
markets = [(20,2), (10,1), (5,1/2)] #three inv. demand functions (int.,slope)
c1 = 2 #cost per unit output for each player
c2 = 2
#inverse demand function for demand node below 
def demand(q1, q2, m): 
    Q = q1+q2
    a = m[0]
    b = m[1]
    price = a-b*Q
    return price
    
###########################################
##MARKET Conditions ChanceNode
###########################################
MCPT = np.ones(len(markets))/len(markets) #CPT for demand function at Market node
Mparents = [] #no parents for market demand because it is a root node
Mspace = markets #the space of the market node
Mip = (MCPT, Mparents, Mspace) #CPT input tuple
M = ChanceNode('M', CPTip=Mip, description='determine market conditions')

###########################################
##ONE DecisionNode FOR EACH PLAYER
###########################################
#note that 1 observes M but not Q2
Q1 = DecisionNode('Q1', '1', actions, [M], 'Stackelberg leader q')
#note that 2 observes Q1 but not M
Q2 = DecisionNode('Q2', '2', actions, [Q1], 'Stackelberg follower q')

###########################################
##DEMAND DeterNode
###########################################
Dfunc = demand #the deterministic function for the Demand DeterNode
Dparams =  {'q1': Q1, 'q2': Q2, 'm': M} #parameter input for Demand DeterNode
Dcont = True    
D = DeterNode('D', Dfunc, Dparams, Dcont, description='inv market demand')

###########################################
##UTILITY FUNCTIONS
###########################################
def util1(Q1,D):
    #PLAYER 1's utility function (profits)
    profit = Q1*D-c1*Q1
    return profit

def util2(Q2,D):
    #PLAYER 2's utility function (profits)
    profit = Q2*D-c2*Q2
    return profit

###########################################
##INITIALIZING the SemiNFG
###########################################
u_funcs = {'1': util1, '2': util2} #setting the utility dictionary    
nodeset = set([M,Q1,Q2,D]) #creating the set of nodes to initialize the net
G = SemiNFG(nodeset, u_funcs) #initializing the ne

G.draw_graph() #visualizing the network

###########################################
##MANIPULATING CPTs
###########################################
#Giving 1 a pure random CPT
G.node_dict['Q1'].randomCPT(mixed=False)
#Giving 2 a uniform mixed CPT
G.node_dict['Q2'].uniformCPT()
#Perturbing 1's CPT with noise=0.5
G.node_dict['Q1'].perturbCPT(0.5, mixed=False)

###########################################
##SAMPLING 
###########################################
#Sample the entire Bayesian Network
G.sample() 
#sample entire net and return a dict of sampled values for nodes named M and Q2
valuedict = G.sample(nodenames=['M', 'Q2']) 
#sample Q1 and all of its descendants
valuedict = G.sample(start=['Q1'])

###########################################
##GETTING VALUES
###########################################
valuedict = G.get_values(nodenames=['Q1', 'D'])

############################################
###PGT INTELLIGENCE ESTIMATION
############################################
##Defining a welfare function on G
#def welfare(G):
#    G.sample()
#    w = G.utility('1')+G.utility('2')
#    return w
#    
##Defining a PGT posterior on iq profiles (dict)    
#def density(iqdict):
#    x = iqdict.values()
#    y = np.power(x,2)
#    z = np.product(y)
#    return z
#
#GG = copy.deepcopy(G)
#S = 50 #number of samples
#X = 10 #number of samples of utility of G in calculating iq
#M = 20 #number of alternative strategies sampled in calculating iq
#noise = .2 #noise in the perturbations of G for MH or MC sampling
#innoise = noise #satisficing distribution noise for iq calculations
#burn = 100 #number of draws to burn for MH
#
#from pynfg.pgtsolutions.intelligence.coordinated import *
#
#tipoff = time.time() #starting a timer
##Importance Samping estimation of PGT posterior
#intelMC, funcoutMC, weightMC = coordinated_MC(GG, S, noise, X, M, \
#                                              innoise=.2, \
#                                              delta=1, \
#                                              integrand=welfare, \
#                                              mix=False, \
#                                              satisfice=GG)
#halftime = time.time()
#print halftime-tipoff
##Metropolis-Hastings estimation of PGT posterior
#intelMH, funcoutMH, densMH = coordinated_MH(GG, S, density, noise, X, M,\
#                                            innoise=.2, \
#                                            delta=1, \
#                                            integrand=welfare, \
#                                            mix=False, \
#                                            satisfice=GG)
#buzzer = time.time()
##Printing elapsed times
#T = halftime-tipoff
#print 'MC took:', T,  'sec., ', T/60, 'min., or', T/3600, 'hr.'
#T = buzzer-halftime
#print 'MH took:', T,  'sec., ', T/60, 'min., or', T/3600, 'hr.'
#
############################################
###PLOTTING PGT RESULTS
############################################
##selecting output into appropriate lists
#MCiqQ1 = [intelMC[s]['1'] for s in xrange(1,S+1)] 
#MHiqQ1 = [intelMH[s]['1'] for s in xrange(1,S+1)]
##creating the importance sampling weights from MC
#MCweight = [density(intelMC[s])/np.prod(weightMC[s].values()) for s in \
#            xrange(1,S+1)] 
##the PGT distribution over intelligence values for player 1
#plt.figure()
#plt.hist(MCiqQ1, normed=True, weights=MCweight, alpha=0.5)
#plt.hist(MHiqQ1[burn::], normed=True, alpha=0.5)
##the PGT distribution over welfare values
#plt.figure()
#plt.hist(funcoutMC.values(), normed=True, weights=MCweight, alpha=0.5)
#plt.hist(funcoutMH.values()[burn::], normed=True, alpha=0.5)