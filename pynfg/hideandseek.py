# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:22:43 2013

@author: James Bono

PyNFG - a Python package for modeling and solving Network Form Games
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
from nodes import *
from seminfg import SemiNFG, iterSemiNFG
import scipy.stats.distributions as randvars
from RLsolutions import *
import PGTsolutions

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
            np.array([[east,0], [0,north]])}
continuousf = False
spaceseek = [np.array([[w,x], [y,z]]) for w in range(east+1) for x in \
            range(north+1) for y in range(east+1) for z in range(north+1)]
F = DeterNode('Froot0', newloc, paramsf, continuousf, space=spaceseek, \
               basename='Froot', time=0)
# Observational noise for player 1, seeker
CPT1 = np.array([0, 0, 0, 0, 1])
par1 = []
space1 = [up, down, left, right, stay]
CPTip1 = (CPT1, par1, space1)
C1 = ChanceNode('C10', CPTip=CPTip1, basename='C1', time=0)
# Observational noise for player 2, hider
CPT2 = np.array([0, 0, 0, 0, 1])
par2 = []
space2 = [up, down, left, right, stay]
CPTip2 = (CPT2, par2, space2)
C2 = ChanceNode('C20', CPTip=CPTip2, basename='C2', time=0)
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
D1 = DecisionNode('D10', 'seeker', [up, down, left, right, stay], \
                    parents=[Fseek], basename='D1', time=0)
# DecisionNode for the hider
D2 = DecisionNode('D20', 'hider', [up, down, left, right, stay], \
                    parents=[Fhide], basename='D2', time=0)

nodeset = set([F,Fseek,Fhide,C1,C2,D1,D2])

paramsf = {'var1': D1, 'var2': D2, 'var3': F}
F = DeterNode('F0', newloc, paramsf, continuousf, space=spaceseek,\
                basename='F', time=0)
nodeset.add(F)

# iteratively building up the net               
for t in range(1,10):
                    
    C1 = ChanceNode('C1%s' %t, CPTip=CPTip1, basename='C1', time=t)
    nodeset.add(C1)
    
    C2 = ChanceNode('C2%s' %t, CPTip=CPTip2, basename='C2', time=t)
    nodeset.add(C2)
                    
    paramsseek = {'var1': C1, 'var2': F}
    Fseek = DeterNode('Fseek%s' %t, adjust1, paramsseek, continuousseek, \
              space=spaceseek, basename='Fseek', time=t)
    nodeset.add(Fseek)
    
    paramshide = {'var1': C2, 'var2': F}
    Fhide = DeterNode('Fhide%s' %t, adjust2, paramshide, continuousseek, \
              space=spaceseek, basename='Fhide', time=t)
    nodeset.add(Fhide)
              
    D1 = DecisionNode('D1%s' %t, 'seeker', [up, down, left, right, stay], \
                    parents=[Fseek], basename='D1', time=t)
    nodeset.add(D1)
                    
    D2 = DecisionNode('D2%s' %t, 'hider', [up, down, left, right, stay], \
                    parents=[Fhide], basename='D2', time=t)
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

G.basename_partition['D2'][0].randomCPT(mixed=False)
for n in G.basename_partition['D2'][1:]:
    n.CPT = G.basename_partition['D2'][0].CPT

G.basename_partition['D1'][0].uniformCPT()
NN = 50
#        
G1, Rseries = ewma_jaakkola(G, 'D1',J=np.floor(linspace(100,10,num=NN)), N=NN, \
                            alpha=1, delta=.8, \
                            eps=linspace(.05,0.2,num=NN))

#G1.sample_timesteps(G1.starttime)
#
#for t in range(G1.starttime, G1.endtime+1):
    