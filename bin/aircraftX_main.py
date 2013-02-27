# -*- coding: utf-8 -*-
"""
A simple iterSemiNFG example of three aircraft avoiding each other en route

Created on Mon Feb 11 14:40:51 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""
from __future__ import division

import time
from numpy.linalg import norm
from math import acos
from classes import *
from aircraftX_utils import *
from rlsolutions import *
from pgtsolutions.intelligence import iq_MC_iter

# PARAMETERS
#number of time steps
T = 10
#action spaces for aircraft
actions = [pi/2, pi/4, 0, -pi/4, -pi/2]
#starting locations
loca = np.array([0,0])
locb = np.array([5,0])
locc = np.array([2.5, 4.33])
#starting vectors
veca = np.array([1,1])/norm(np.array([1,1]))
vecb = np.array([-1,1])/norm(np.array([-1,1]))
vecc = np.array([0,-1])
#locations of terminal airports
goal = [np.array([5,5]), np.array([0,5]), np.array([2.5,4.33-7.07])]
#Euclidian distance covered per time step by each aircraft
speed = [.5, .5, .5]
#defining safe distances from aircraft
redzone = .5
orangezone = 1
#penalties for not maintaining safe distance
redpen = -500
orangepen = -25
#defining distance markers from terminal airport
termzone = 1.5 #getting close
landzone = .5 #within this dist., aircraft are considered landed
#reward for getting close/landing
termrew = 25 
landrew = 50

# FUNCTIONS USED
def frootfunc(locvec):
    #a dummy function that just spits out the starting loc & vec
    return locvec

def rotmat(angle):
    #returns the rotation matrix for a given angle
    r = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    return r

def observe(locvec, p):
    loc = locvec[0]
    vec = locvec[1]
    dist = [norm(loc[p]-x) for x in loc] #list of distances from p
    dist[p] = 3000 #change from 0 to big number so argmin is an opponent
    opp = argmin(dist) #index of opponent
    diff = loc[opp]-loc[p] #location - relative to p - of nearest opponent
    oppquad = get_quad(diff, vec[p])
    #distance to opponent
    if dist[opp]<=redzone: #within redzone
        oppdist = 1 
    elif dist[opp]<=orangezone: #within orangezone
        oppdist = 2
    else: #in safezone
        oppdist = 3
    #angle to airport
    diff = goal[p]-loc[p]
    goalquad = get_quad(diff, vec[p])
    #distance to airport
    gdist = norm(diff)
    if gdist<=termzone: #within terminal zone
        goaldist = 1
    else: #outside terminal zone
        goaldist = 2
    return [oppquad, oppdist, goalquad, goaldist]

def get_quad(diff, vec):
    invtheta = round(dot(diff,vec)/(norm(diff)), 10)
    if abs(invtheta)>1:
        raise ValueError('diff: %s, and vec: %s' %(diff,vec))
    else:
        ang = acos(invtheta) #angle between airport and vec[p]
    if ang<pi/4: #straight ahead
        quad = 0
    elif norm(dot(rotmat(ang),vec)-diff)<norm(dot(rotmat(-ang),vec)-diff):
        if ang<pi/2: #NE orthant
            quad = 1
        else: #NW orthant
            quad = 2
    else:
        if ang<pi/2: #SE orthant
            quad = 4
        else: #SW orthant
            quad = 3
    return quad

def updateloc(acta,actb,actc, locvec=[[loca, locb, locc],[veca, vecb, vecc]]):
    #updates loc and vec according to act
    loc = locvec[0]
    vec = locvec[1]
    act = [acta, actb, actc]
    newloc = []
    newvec = []
    for p in range(len(loc)):
        if norm(loc[p]-goal[p]) <= landzone: #no change to loc or vec
            newloc.append(loc[p])
            newvec.append(vec[p])
        else: #still en route, so calculate new loc and vec
            newloc.append(loc[p]+speed[p]*dot(rotmat(act[p]),vec[p]))
            newvec.append((newloc[p]-loc[p])/norm(newloc[p]-loc[p]))
    return [newloc, newvec]
    
# THE NODES
paramsf = {'locvec': [[loca, locb, locc], [veca, vecb, vecc]]}
continuousf = True
Fr = DeterNode('Froot0', frootfunc, paramsf, continuousf, basename='Froot', \
                time=0)

paramsfa = {'locvec': Fr, 'p': 0}
continuousfa = False
spacefa = [[w,x,y,z] for w in range(5) for x in [1,2,3] \
            for y in range(5) for z in [1,2]]
FA = DeterNode('FA0', observe, paramsfa, continuousfa, space=spacefa, \
                basename='FA', time=0)
                
paramsfb = {'locvec': Fr, 'p': 1}
FB = DeterNode('FB0', observe, paramsfb, continuousfa, space=spacefa, \
                basename='FB', time=0)

paramsfc = {'locvec': Fr, 'p': 2}
FC = DeterNode('FC0', observe, paramsfc, continuousfa, space=spacefa, \
                basename='FC', time=0)

DA = DecisionNode('DA0', 'A', actions, parents=[FA], basename='DA', time=0)
DB = DecisionNode('DB0', 'B', actions, parents=[FB], basename='DB', time=0)
DC = DecisionNode('DC0', 'C', actions, parents=[FC], basename='DC', time=0)

paramsf = {'locvec': Fr, 'acta': DA, 'actb': DB, 'actc': DC}
continuousf = True
F = DeterNode('F0', updateloc, paramsf, continuousf, basename='F', \
                time=0)
#collecting nodes in a set
nodes = set([Fr,FA,FB,FC,DA,DB,DC,F])

# BUILDING THE NET
for t in range(1,T):
    
    paramsfa = {'locvec': F, 'p': 0}
    FA = DeterNode('FA%s' %t, observe, paramsfa, continuousfa, space=spacefa, \
                    basename='FA', time=t)
                    
    paramsfb = {'locvec': F, 'p': 1}
    FB = DeterNode('FB%s' %t, observe, paramsfb, continuousfa, space=spacefa, \
                    basename='FB', time=t)
    
    paramsfc = {'locvec': F, 'p': 2}
    FC = DeterNode('FC%s' %t, observe, paramsfc, continuousfa, space=spacefa, \
                    basename='FC', time=t)
    
    DA = DecisionNode('DA%s' %t, 'A', actions, parents=[FA], basename='DA', time=t)
    DB = DecisionNode('DB%s' %t, 'B', actions, parents=[FB], basename='DB', time=t)
    DC = DecisionNode('DC%s' %t, 'C', actions, parents=[FC], basename='DC', time=t)
    
    paramsf = {'locvec': F, 'acta': DA, 'actb': DB, 'actc': DC}
    F = DeterNode('F%s' %t, updateloc, paramsf, continuousf, basename='F', \
                    time=t)
    nodes.update([FA,FB,FC,DA,DB,DC,F])#updating the node set

# REWARD FUNCTIONS
def distrew(goaldist, oppdist):
    if oppdist>orangezone or goaldist<landzone: #no penalty assessed in landzone
        pen = 0
    elif oppdist<redzone: 
        pen = redpen
    else:
        pen = orangepen
    if goaldist<landzone:
        rew = landrew
    elif goaldist<termzone:
        rew = termrew
    else:
        rew = 0
    return rew+pen
#A's reward                    
def rewardA(F):
    loc = F[0]
    goaldist = norm(loc[0]-goal[0])
    dist = [norm(loc[0]-x) for x in loc] #list of distances from p
    dist[0] = 3000 #change from 0 to big number so argmin is an opponent
    oppdist = min(dist) 
    return distrew(goaldist, oppdist)
#B's reward    
def rewardB(F):
    loc = F[0]
    goaldist = norm(loc[1]-goal[1])
    dist = [norm(loc[1]-x) for x in loc] #list of distances from p
    dist[1] = 3000 #change from 0 to big number so argmin is an opponent
    oppdist = min(dist) 
    return distrew(goaldist, oppdist)
#C's reward
def rewardC(F):
    loc = F[0]
    goaldist = norm(loc[2]-goal[2])
    dist = [norm(loc[2]-x) for x in loc] #list of distances from p
    dist[2] = 3000 #change from 0 to big number so argmin is an opponent
    oppdist = min(dist) 
    return distrew(goaldist, oppdist)
#player-keyed dictionary of rewards
r_funcs = {'A': rewardA, 'B': rewardB, 'C': rewardC}
#initializing the iterSemiNFG    
G = iterSemiNFG(nodes, r_funcs)
#G.draw_graph()
##setting A, B and C to level 0.
#G.bn_part['DA'][0].CPT = calc_level0(G.bn_part['DA'][0])
#G.bn_part['DB'][0].CPT = calc_level0(G.bn_part['DB'][0])
#G.bn_part['DC'][0].CPT = calc_level0(G.bn_part['DC'][0])
#for t in range(G.starttime, G.endtime+1):
#    G.bn_part['DA'][t].CPT = G.bn_part['DA'][0].CPT
#    G.bn_part['DB'][t].CPT = G.bn_part['DB'][0].CPT
#    G.bn_part['DC'][t].CPT = G.bn_part['DC'][0].CPT
##perturbing DA for the MC RL training
#G.bn_part['DB'][0].perturbCPT(0.2, mixed=True)
#
#G1, returnfig = ewma_mcrl(G, 'DB', 40, 100, .7, 1, 0.1, pureout=False)
#
#adict = G1.sample_timesteps(G1.starttime, basenames=['F'])
#routefig = plotroutes(adict['F'], [loca, locb, locc], goal)
#find_collisions(G1, redpen, orangepen, verbose=True)
#go = time.time()
#intel, funcout = iq_MH_iter(G, 10, 1, 10, 1)
#print time.time()-go