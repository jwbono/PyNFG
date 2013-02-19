# -*- coding: utf-8 -*-
"""
Calculating level-0 policies for 3aircraft_main.py

Created on Wed Feb 13 14:47:10 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""
import numpy as np
from classes import *
from matplotlib import pylab

def calc_level0(DN):
    CPT = np.zeros(DN.CPT.shape)
    head={0: 2, 1: 1, 2: 0, 3: 4, 4: 3} #points the aircraft towards goal
    for i in range(5):# for each goalquad, optimal direction gets prob. 1
        #note that y plays the role of goalquad
        spacefa = [[w,x,y,z] for w in range(5) for x in [1,2,3] \
                                        for y in range(i,i+1) for z in [1,2]]
        for x in spacefa:
            ind = DN.get_CPTindex([x], onlyparents=True)
            indo = list(ind)
            indo.append(head[x[2]])
            ind = tuple(indo)
            CPT[ind] = 1
    return CPT
    
def plotroutes(routes, origin, goal):
    #routes is the output of G.sample_timesteps(0, basename=['F'])-->adict['F']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    nplayers = len(routes[0][0])
    x, y = org_routes(routes)
    for i in range(nplayers):
        pylab.plot(x[i],y[i], colors[i]+'--')
        pylab.plot(goal[i][0],goal[i][1], colors[i]+'x') #goal marked with x
        pylab.plot(origin[i][0], origin[i][1], colors[i]+'o') #origin marked with o
    x = pylab.gcf()
    pylab.show()
    return x
        
def org_routes(routes):
    #routes is the output of G.sample_timesteps(0, basename=['F'])-->adict['F']
    nplayers = len(routes[0][0])
    x = {}
    y = {}
    for i in range(nplayers):# getting lists of x and y coords for each player
        x[i] = [routes[t][0][i][0] for t in range(len(routes))]
        y[i] = [routes[t][0][i][1] for t in range(len(routes))]
    return x, y
    
def find_collisions(G, collpen, closepen, verbose=False):
    collisions = {}
    closecalls = {}
    for p in G.players:
        collisions[p] = []
        closecalls[p] = []
        for t in range(G.endtime):
            if G.reward(p, t)<=collpen:
                collisions[p].append(t)
            elif G.reward(p, t)<=closepen:
                closecalls[p].append(t)
        print p, 'had collisions at times: ', collisions[p]
        print p, 'had closecalls at times: ', closecalls[p]
    return collisions, closecalls
        