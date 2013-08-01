# -*- coding: utf-8 -*-
"""
Implements a simple hide-and-seek iterSemiNFG:
There is a hider and a seeker. In each time step, each player is at a location
on a grid. The seeker gets a point each time both players are at the same 
location. The hider loses a point when this happens. In each time step, players
each make an observation (possibly noisy) about their opponent's location and 
choose one of the five following moves: up, down, left, right, stay. These moves
result in an associated location change of one grid step. A player on the 
"northern" boundary that chooses up remains as is. Similiar rules apply to 
other moves that would result in locations off the grid.

Note: It is better to run this script line by line or customize your own run
script from the pieces contained herein rather than running the entire file. The
reason is that the PGT and RL algorithms will take a long time.

Created on Mon Jan 28 16:22:43 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""
from __future__ import division

from pynfg import DecisionNode, DeterNode, ChanceNode
from pynfg import SemiNFG, iterSemiNFG
import numpy as np
import scipy.stats.distributions as randvars
import time
import copy

###########################################
##PARAMETERS AND FUNCTIONS
###########################################
# boundaries of the grid
west = 0
east = 2
north = 2
south = 0

# actions of the players
up = np.array([0,1])
down = np.array([0,-1])
left = np.array([-1,0])
right = np.array([1,0])
stay = np.array([0,0])
# space of actions that players can choose
actionspace = [up, down, left, right, stay] 

# time steps
T = 10

# starting locations
startingloc = np.array([[east,north-1], [0,north-1]])

# observational noise CPT (up, down, left, right, stay)
obsnoiseCPT = np.array([.1, .1, .1, .1, .6])

# the space for the state nodes below, Froot and F
statespace = [np.array([[w,x],[y,z]]) \
                for w in range(east+1) for x in range(north+1) \
                for y in range(east+1) for z in range(north+1)]

# a function that adjusts for moves off the grid
def adjust_loc(location):
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
def newloc(seekmove=np.array([0,0]), hidemove=np.array([0,0]), \
            loc=startingloc):
    locseek = adjust_loc(seekmove+loc[0])
    lochide = adjust_loc(hidemove+loc[1])
    return np.vstack((locseek, lochide))

# Combines observational noise and location to give a valid location on-grid
# for seeker's observation of hider's location
def adjust_seeker(noise, loc):
    opponent = adjust_loc(noise+loc[1]) 
    return np.vstack((loc[0], opponent))

# Combines observational noise and location to give a valid location on-grid.
# for hider's observation of seeker's location
def adjust_hider(noise, loc):
    opponent = adjust_loc(noise+loc[0])
    return np.vstack((opponent, loc[1]))                 

##################################################
##THE NODES - first time 0, then 1 to T in a loop
##################################################
#STATE ROOT NODE, DeterNode F 
paramsf = {}
continuousf = False
Froot = DeterNode('Froot0', newloc, paramsf, continuousf, space=statespace, \
               basename='Froot', time=0)
               
#OBSERVATIONAL NOISE for SEEKER, ChanceNode Cseek
parseek = []
CPTipseek = (obsnoiseCPT, parseek, actionspace)
Cseek = ChanceNode('Cseek0', CPTip=CPTipseek, basename='Cseek', time=0)

#OBSERVATIONAL NOISE for HIDER, ChanceNode C2
parhide = []
CPTiphide = (obsnoiseCPT, parhide, actionspace)
Chide = ChanceNode('Chide0', CPTip=CPTiphide, basename='Chide', time=0)

#COMBINE OBS NOISE FOR SEEKER, DeterNode Fseek
paramsseek = {'noise': Cseek, 'loc': Froot}
continuousseek = False
Fseek = DeterNode('Fseek0', adjust_seeker, paramsseek, continuousseek, \
                  space=statespace, basename='Fseek', time=0)
#COMBINE OBS NOISE FOR SEEKER, DeterNode Fseek
paramshide = {'noise': Chide, 'loc': Froot}
Fhide = DeterNode('Fhide0', adjust_hider, paramshide, continuousseek, \
                  space=statespace, basename='Fhide', time=0)
                  
#SEEKER DecisionNode, Dseek
Dseek = DecisionNode('Dseek0', 'seeker', actionspace, parents=[Fseek], \
                     basename='Dseek', time=0)
#HIDER DecisionNode, Dhide
Dhide = DecisionNode('Dhide0', 'hider', actionspace, parents=[Fhide], \
                     basename='Dhide', time=0)
                     
#STATE ROOT NODE, DeterNode F 
paramsf = {'seekmove': Dseek, 'hidemove': Dhide, 'loc': Froot}
F = DeterNode('F0', newloc, paramsf, continuousf, space=statespace, \
              basename='F', time=0)
                
#adding time 0 nodes to nodeset
nodeset = set([F,Froot,Fseek,Fhide,Cseek,Chide,Dseek,Dhide])

#BUILD time steps 1 to T-1 iteratively               
for t in range(1,T):
                    
    Cseek = ChanceNode('Cseek%s' %t, CPTip=CPTipseek, basename='Cseek', time=t)
    
    Chide = ChanceNode('Chide%s' %t, CPTip=CPTiphide, basename='Chide', time=t)
                    
    paramsseek = {'noise': Cseek, 'loc': F}
    Fseek = DeterNode('Fseek%s' %t, adjust_seeker, paramsseek, continuousseek, \
                      space=statespace, basename='Fseek', time=t)
    
    paramshide = {'noise': Chide, 'loc': F}
    Fhide = DeterNode('Fhide%s' %t, adjust_hider, paramshide, continuousseek, \
                      space=statespace, basename='Fhide', time=t)
              
    Dseek = DecisionNode('Dseek%s' %t, 'seeker', actionspace, \
                         parents=[Fseek], basename='Dseek', time=t)
                    
    Dhide = DecisionNode('Dhide%s' %t, 'hider', actionspace, \
                         parents=[Fhide], basename='Dhide', time=t)
    
    paramsf = {'seekmove': Dseek, 'hidemove': Dhide, 'loc': F}
    F = DeterNode('F%s' %t, newloc, paramsf, continuousf, space=statespace, \
                  basename='F', time=t)
    #adding time t nodes to nodeset             
    nodeset.update([F,Fseek,Fhide,Cseek,Chide,Dseek,Dhide])

##########################
##REWARD FUNCTIONS
##########################
# seeker's reward function    
def seek_reward(F):
    if np.array_equal(F[0], F[1]):
        return 1
    else:
        return 0
# hider's reward function    
def hide_reward(F):
    return -1*seek_reward(F)

#rewards dictionary
rfuncs = {'seeker': seek_reward, 'hider': hide_reward}

##################################
##CREATING THE iterSemiNFG
##################################
G = iterSemiNFG(nodeset, rfuncs)

#making a set of the names of the first two time steps for visualization
drawset = set([n.name for n in G.time_partition[0]]).union(set([n.name for \
                                                    n in G.time_partition[1]]))
G.draw_graph(drawset) #visualizing the first two time steps of the net

###########################################
##MANIPULATING CPTs
###########################################
#Giving hider a uniform CPT
G.bn_part['Dhide'][0].uniformCPT()
#Giving seeker a pure random CPT
G.bn_part['Dseek'][0].randomCPT(mixed=False)
#pointing all CPTs to time 0 CPT
cptdict = G.get_decisionCPTs(mode='basename')
G.set_CPTs(cptdict)

###########################################
##SAMPLING 
###########################################
#Sample the entire Bayesian Network
G.sample()
#sample entire net and return a dict of sampled values for node Dhide8 and F1
valuedict = G.sample(nodenames=['Dhide8', 'F1'])
#Sample timesteps 3 through 6 and returning a dict with values for specific basenames
valuedict = G.sample_timesteps(3, 6, basenames=['Dhide', 'F', 'Cseek'])  
#sample F4 and all of its descendants
G.sample(start=['F4'])

###########################################
##GETTING VALUES
###########################################
valuedict = G.get_values(nodenames=['Cseek0', 'Dhide8'])

#####################################################
##TRAINING LEVEL 1 with ewma_mcrl
#####################################################
from pynfg.levelksolutions.mcrl import *

N=10
GG = copy.deepcopy(G) #NOTE: setting uni=True below starts Dseek as uniform
#Train Seeker against L0 Hider
GseekL1, returnfig = mcrl_ewma(GG, 'Dseek', np.linspace(50,1,N), N, \
                              np.linspace(.5,1,N), 1, np.linspace(.2,1,N), \
                              uni=True, pureout=True)
GG = copy.deepcopy(G) #NOTE: setting uni=True below starts Dseek as uniform
#Train Hider against L0 Seeker
GhideL1, returnfig = mcrl_ewma(GG, 'Dhide', np.linspace(50,1,N), N, \
                              np.linspace(.5,1,N), 1, np.linspace(.2,1,N), \
                              uni=True, pureout=True)
#Create G1 from GhideL1 by "importing" L1 seeker's CPT
G1 = copy.deepcopy(GhideL1)
for n in GhideL1.bn_part['Dseek']:
    n.CPT = GseekL1.bn_part['Dseek'][0].CPT

############################################
###PGT INTELLIGENCE ESTIMATION
############################################
#defining a welfare metric on G
def captures(G):
    T0 = G.starttime
    T = G.endtime
    G.sample()
    num_captures = G.npv_reward('seeker', T0, 1)
    return num_captures/(T-T0)

#Defining a PGT posterior on iq profiles (dict) 
def density(iq):
    x = iq.values()
    y = np.power(x, 2)
    z = np.prod(y)
    return z
    
GG = copy.deepcopy(G1) #NOTE: the CPTs of G are seeds for MH and MC sampling
S = 30 #number of samples
X = 10 #number of samples of utility of G in calculating iq
M = 20 #number of alternative strategies sampled in calculating iq
noise = .2 #noise in the perturbations of G for MH or MC sampling
innoise = noise #satisficing distribution noise for iq calculations
burn = 10 #number of draws to burn for MH

from pynfg.pgtsolutions.intelligence.policy import *

tipoff = time.time() #starting a timer
#Importance Samping estimation of PGT posterior
intelMC, funcoutMC, weightMC = policy_MC(GG, S, noise, X, M, \
                                        innoise=.2, \
                                        delta=1, \
                                        integrand=captures, \
                                        mix=False, \
                                        satisfice=GG)
halftime = time.time()
print halftime-tipoff
#Metropolis-Hastings estimation of PGT posterior
intelMH, funcoutMH, densMH = policy_MH(GG, S, density, noise, X, M, \
                                       innoise=.2, \
                                       delta=1, \
                                       integrand=captures, \
                                       mix=False, \
                                       satisfice=GG)
buzzer = time.time()
#Printing elapsed times
T = halftime-tipoff
print 'MC took:', T,  'sec., ', T/60, 'min., or', T/3600, 'hr.'
T = buzzer-halftime
print 'MH took:', T,  'sec., ', T/60, 'min., or', T/3600, 'hr.'

###########################################
##PLOTTING PGT RESULTS
###########################################
#creating the importance sampling weights from MC
MCweight = [density(intelMC[s])/np.prod(weightMC[s].values()) for s in \
            xrange(1,S+1)]
#the PGT distributions over welfare values
plt.figure()
plt.hist(funcoutMC.values(), normed=True, weights=MCweight, alpha=.5) 
plt.hist(funcoutMH.values()[burn::], normed=True, alpha=.5)
plt.show()




    