# -*- coding: utf-8 -*-
"""
Implements a simple hide-and-seek iterSemiNFG

Part of: PyNFG - a Python package for modeling and solving Network Form Games

Created on Mon Jan 28 16:22:43 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""
from __future__ import division

from pynfg import DecisionNode, DeterNode, ChanceNode
from pynfg import SemiNFG, iterSemiNFG
import numpy as np
import scipy.stats.distributions as randvars
from pynfg.rlsolutions.mcrl import *
from pynfg.pgtsolutions.intelligence.iq_coord import *
import time
import copy

# boundaries of the grid
west = 0
east = 2
north = 2
south = 0

# moves of the players
up = np.array([0,1])
down = np.array([0,-1])
left = np.array([-1,0])
right = np.array([1,0])
stay = np.array([0,0])

#time steps
T = 10

# a function that adjusts for moves off the grid
def adjust(location):
    if location[0]<west:
        location[0] = west
    elif location[0]>east:
        location[0] = east
        
    if location[1]<south:
        location[1] = south
    elif location[1]>north:
        location[1] = north
        
    return location

# Combines start loc. and moves to give new loc. Used by DeterNode F
def newloc(var1=np.array([0,0]), var2=np.array([0,0]), \
            var3=np.array([[east,0], [0,north]])):
    location1 = adjust(var1+var3[0])
    location2 = adjust(var2+var3[1])
    return [location1, location2]                

# root DeterNode F
paramsf = {'var1': np.array([0,0]), 'var2': np.array([0,0]), 'var3': \
            np.array([[east,north-1], [0,north-1]])}
continuousf = False
spaceseek = [np.array([[w,x], [y,z]]) for w in range(east+1) for x in \
            range(north+1) for y in range(east+1) for z in range(north+1)]
F = DeterNode('Froot0', newloc, paramsf, continuousf, space=spaceseek, \
               basename='Froot', time=0)
# Observational noise for player 1, seeker
CPT1 = np.array([.1, .1, .1, .1, .6])
par1 = []
space1 = [up, down, left, right, stay]
CPTip1 = (CPT1, par1, space1)
C1 = ChanceNode('Cseek0', CPTip=CPTip1, basename='Cseek', time=0)
# Observational noise for player 2, hider
CPT2 = CPT1
par2 = []
space2 = [up, down, left, right, stay]
CPTip2 = (CPT2, par2, space2)
C2 = ChanceNode('Chide0', CPTip=CPTip2, basename='Chide', time=0)
# Combines observational noise and location to give a valid location on-grid.
def adjust1(var1, var2):
    opponent = adjust(var1+var2[1]) 
    return [var2[0], opponent]                   
# DeterNode combines obs noise and location for the seeker
paramsseek = {'var1': C1, 'var2': F}
continuousseek = False
Fseek = DeterNode('Fseek0', adjust1, paramsseek, continuousseek, \
              space=spaceseek, basename='Fseek', time=0)
# Combines observational noise and location to give a valid location on-grid.
def adjust2(var1, var2):
    opponent = adjust(var1+var2[0])
    return [opponent, var2[1]] 
# DeterNode combines obs noise and location for the hider  
paramshide = {'var1': C2, 'var2': F}
Fhide = DeterNode('Fhide0', adjust2, paramshide, continuousseek,
              space=spaceseek, basename='Fhide', time=0)
# DecisionNode for the seeker
D1 = DecisionNode('Dseek0', 'seeker', [up, down, left, right, stay], \
                    parents=[Fseek], basename='Dseek', time=0)
# DecisionNode for the hider
D2 = DecisionNode('Dhide0', 'hider', [up, down, left, right, stay], \
                    parents=[Fhide], basename='Dhide', time=0)

nodeset = set([F,Fseek,Fhide,C1,C2,D1,D2])

paramsf = {'var1': D1, 'var2': D2, 'var3': F}
F = DeterNode('F0', newloc, paramsf, continuousf, space=spaceseek,\
                basename='F', time=0)
nodeset.add(F)

# iteratively building up the net               
for t in range(1,T+1):
                    
    C1 = ChanceNode('Cseek%s' %t, CPTip=CPTip1, basename='Cseek', time=t)
    nodeset.add(C1)
    
    C2 = ChanceNode('Chide%s' %t, CPTip=CPTip2, basename='Chide', time=t)
    nodeset.add(C2)
                    
    paramsseek = {'var1': C1, 'var2': F}
    Fseek = DeterNode('Fseek%s' %t, adjust1, paramsseek, continuousseek, \
              space=spaceseek, basename='Fseek', time=t)
    nodeset.add(Fseek)
    
    paramshide = {'var1': C2, 'var2': F}
    Fhide = DeterNode('Fhide%s' %t, adjust2, paramshide, continuousseek, \
              space=spaceseek, basename='Fhide', time=t)
    nodeset.add(Fhide)
              
    D1 = DecisionNode('Dseek%s' %t, 'seeker', [up, down, left, right, stay], \
                    parents=[Fseek], basename='Dseek', time=t)
    nodeset.add(D1)
                    
    D2 = DecisionNode('Dhide%s' %t, 'hider', [up, down, left, right, stay], \
                    parents=[Fhide], basename='Dhide', time=t)
    nodeset.add(D2)
    
    paramsf = {'var1': D1, 'var2': D2, 'var3': F}
    F = DeterNode('F%s' %t, newloc, paramsf, continuousf, space=spaceseek,\
                basename='F', time=t)
    nodeset.add(F)
# seeker's reward function    
def reward1(F=np.array([[east,0],[0,north]])):
    if np.array_equal(F[0], F[1]):
        return 1
    else:
        return 0
# hider's reward function    
def reward2(F=np.array([[1,0],[0,1]])):
    return -1*reward1(F)

rfuncs = {'seeker': reward1, 'hider': reward2}
G = iterSemiNFG(nodeset, rfuncs)

G.bn_part['Dhide'][0].uniformCPT()
G.bn_part['Dseek'][0].randomCPT()
for t in xrange(1, G.endtime+1):
    G.bn_part['Dhide'][t].CPT = copy.copy(G.bn_part['Dhide'][0].CPT)
    G.bn_part['Dseek'][t].CPT = copy.copy(G.bn_part['Dseek'][0].CPT)

#drawset = set(G.time_partition[0]).union(set(G.time_partition[1]))
#G.draw_graph(drawset)

def density(iq):
    x = iq.values()
    y = np.power(x, 2)
    z = np.prod(y)
    return z

def captures(G):
    T0 = G.starttime
    T = G.endtime
    G.sample()
    num_captures = G.npv_reward('seeker', T0, 1)
    return num_captures/(T-T0)

S = 1000
X = 10
M = 30
delta = 1
noise = .1
burn = 200
go = time.time()

#intelMC, funcoutMC, weightMC = iq_MC_coord(G, S, noise, X, M, innoise=1, \
#                                                            integrand=captures)
#weightlist = np.array([weightMC[s]['hider']**-1 for s in xrange(1, S+1)])                                                           
#probMC = weightlist/np.sum(weightlist)
#
#iqhiderMC = [intelMC[s]['hider'] for s in xrange(1,S+1)]
#plt.figure()
#plt.hist(iqhiderMC, normed=True, weights=probMC)
#
#social_welfare = [funcoutMC[s] for s in xrange(1,S+1)]
#plt.figure()
#plt.hist(social_welfare, normed=True, weights=probMC) 
#

#intelMH, funcoutMH, densMH = iq_MH_coord(G, S, density, noise, X, M, \
#                                                innoise=.4, integrand=captures)
#                                                
#iqhiderMH = [intelMH[s]['hider'] for s in xrange(1,S+1)]
#weightMH = densMH[burn::]
#plt.figure()
#plt.hist(iqhiderMH[burn::], normed=True, weights=weightMH)
#
#social_welfare = [funcoutMH[s] for s in xrange(1,S+1)]
#plt.figure()
#plt.hist(social_welfare[burn::], normed=True, weights=weightMH)
#
N=60

G1, returnfig = ewma_mcrl(copy.deepcopy(G), 'Dseek', np.linspace(50,1,N), N, \
                        np.linspace(.5,1,N), 1, np.linspace(.2,1,N), uni=True, \
                        pureout=True)
print (time.time()-go)/60
#captures(G1)
#G1.sample_timesteps(G1.starttime)
#
#for t in range(G1.starttime, G1.endtime+1):
    