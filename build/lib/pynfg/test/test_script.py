# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:56:01 2012

@author: jamesbono

Here's a test script for use in developing the PyNFG library
"""
import numpy as np
from nodes import *
from seminfg import SemiNFG, iterSemiNFG
import scipy.stats.distributions as randvars
from RLsolutions import *
import PGTsolutions

#D1 = DecisionNode('D1', '1', ['a', 'b'], parents=None, description='root')
#D2 = DecisionNode('D2', '2', ['c', 'd', 'e'], parents=[D1], \
#                    description='This is a child node of D1')
#D3 = DecisionNode('D3', '1', ['f', 'g', 'h', 'i'], parents=[D1, D2], \
#                    description='This is a child node of D1 and D2')
#
#D1.CPT = D1.randomCPT(mixed=True)
#D2.CPT = D2.randomCPT(mixed=True)
#D3.CPT = D3.randomCPT(mixed=True)
#
#print( "The probability that D1 chooses %s is %s" % \
#        (D1.value, str(D1.prob())))
#
#parinput2 = {'D1': 'a'}
#print( "The probability that D2 chooses %s given D1 chooses %s is %s" \
#        % (D2.value, parinput2['D1'], str(D2.prob(parentinput=parinput2))) )
#        
#noise = 0.1
#D2.CPT = D2.perturbCPT(noise)
#print( "After perturbing D2.CPT with noise=%s, the new probability is %s"\
#        % (str(noise), str(D2.prob(parentinput=parinput2))) )
#        
#parinput3 = {'D1': 'b', 'D2': 'c'}
#print( "The probability that D3 chooses %s given D1 chooses %s and D2 chooses"\
#        " %s is %s" % (D3.value, parinput3['D1'], parinput3['D2'], \
#        str(D2.prob(parentinput=parinput3))) )
#        
#print("Comparing D2.CPT before and after perturbing a sliver")
#print(D2.CPT[0,:])
#D2.CPT = D2.perturbCPT(0.1, sliver=parinput2)        
#print(D2.CPT[0,:])
#
#print("Comparing D3.CPT before and after perturbing a sliver")
#print(D3.CPT[1,0,:])
#D3.CPT = D3.perturbCPT(0.1, sliver=parinput3)        
#print(D3.CPT[1,0,:])
#
#dist1 = randvars.randint
#params1 = [1, 4]
#space1 = [1, 2, 3]
#distip1 = (dist1, params1, space1)
#C1 = ChanceNode('C1', description='root CN given by randint 1 to 4', \
#                distip=distip1)
#
#CPT2 = np.array([[0.25, 0.75], [0.66, 0.34], [0.75, 0.25]])
#parents2 = [C1]
#space2 = [1, 10]
#CPTip2 = (CPT2, parents2, space2)
#C2 = ChanceNode('C2', description='CPT with values either 1 or 2', \
#                CPTip = CPTip2)
#
#dist3 = randvars.norm
#params3 = [C2, 1]
#distip3 = (dist3, params3)
#C3 = ChanceNode('C3', description='leaf CN given by norm with scale fixed and '\
#                +'loc given by C2', distip = distip3)
#
#print('Prob of %s at C1: %s' % (C1.value, C1.prob()))
#print('Prob of %s at C2 given C1 is %s: %s' % (C2.value, C1.value, \
#        C2.prob()))
#print('Prob of %s at C3 given C2 is %s: %s' % (C3.value, C2.value, \
#        C3.prob()))
#
#newc2 = C2.draw_value(parentinput={'C1': 3})    
#
#print newc2
#C2.set_value(newc2)
#print C2.value
#print('Prob of %s at C3 is now: %s' % (C3.value, \
#        C3.prob({'C2': C2.value})))

#dist1 = randvars.randint
#params1 = [1, 4]
#space1 = [1, 2, 3]
#distip1 = (dist1, params1, space1)
#C1 = ChanceNode('C1', description='root CN given by randint 1 to 4', \
#                distip=distip1)
#                
#D1 = DecisionNode('D1', '1', [-1, 0, 1], parents=[C1], \
#                    description='This is a child node of C1')
#                    
#D1.CPT = D1.randomCPT(mixed=False)
#
#dist2 = randvars.norm
#params2 = [D1, 2]
#distip2 = (dist2, params2)
#C2 = ChanceNode('C2', distip=distip2, \
#                description='CN given by norm with scale fixed and loc given by D1')
#        
#def func3(var1=1, var2=0):
#    r = np.sign(var1+var2)
#    return r
#    
#params3 = {'var1': C2}
#continuous3 = False
#space3 = [-1, 0, 1]
#C3 = DeterNode('C3', func3, params3, continuous3, space=space3, \
#                description='a disc. deterministic node child of C2')
#                
#D2 = DecisionNode('D2', '2', [0, 1], parents=[D1, C3], \
#                    description='child of D1 and C3')
#                    
#D2.CPT = D2.randomCPT(mixed=True)
#
#D3 = DecisionNode('D3', '2', [2, 3, 4], parents=[], description='a root DN')
#
#D3.CPT = D3.randomCPT(mixed=False)
#
#def func4(var1=1, var2=3):
#    r = np.exp(var2-var1)
#    return r
#    
#params4 = {'var1': D2, 'var2': D3}
#continuous4 = True
#C4 = DeterNode('C4', func4, params4, continuous4, \
#                description='a cont. deterministic node child of D2 and D3')
#                
#dist5 = randvars.alpha
#params5 = [C4, 2, 6]
#distip5 = (dist5, params5)
#C5 = ChanceNode('C5', distip = distip5, \
#                description='CN given by alpha with shape C4 and loc and scale fixed')
#
#nodeset = set([C1, D1, C2, C3, D2, D3, C4, C5])
#Game = SemiNFG(nodeset)
#Game.sample_prior()
#for x in nodeset:
#    print x.name, x.prob()
#Game.loglike()
#Game.draw_graph()

#from types import IntType
#            
#dist1 = randvars.randint
#params1 = [0, 3]
#space1 = range(3)
#distip1 = (dist1, params1, space1)
#C1 = ChanceNode('C1', distip=distip1, \
#                description='root CN randint from 0 to 2')
#
#D1 = DecisionNode('D1', '1', [0, 1], parents=[C1],\
#                    description='child node of C1. belongs to p1')
#                    
#D2 = DecisionNode('D2', '2', [0, 1], parents=[C1],\
#                    description='child node of C1. belongs to p2')
#
#def funcf(var1, var2, var3):
#    if (var1+var2+var3)%2 == 0:
#        return 'even'
#    else:
#        return 'odd'                    
#
#paramsf = {'var1': D1, 'var2': D2, 'var3': C1}
#continuousf = False
#spacef = ['even', 'odd']
#F1 = DeterNode('F1', funcf, paramsf, continuousf, space=spacef,\
#               description='a disc. DeterNode child of D1, D2, C1')
#               
#def ufunc1(F1):
#    if F1 is 'even':
#        return 1
#    else:
#        return -1
#        
#def ufunc2(F1):
#    if F1 is 'odd':
#        return 1
#    else:
#        return -1
#
#ufuncs = {'1': ufunc1, '2': ufunc2}
#nodeset = set([C1, D1, D2, F1])
#G = SemiNFG(nodeset, ufuncs)
#D2.CPT = D2.randomCPT()
#D1.CPT = D1.randomCPT()
#G.sample(start=[C1])
#G.utility('1')
#G.utility('2')

west = 0
east = 2
north = 2
south = 0

up = np.array([0,1])
down = np.array([0,-1])
left = np.array([-1,0])
right = np.array([1,0])
stay = np.arra([0,0])

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

def Fadjust(var1,var2,var3):
    location1 = adjust(var1+var3[0])
    location2 = adjust(var2+var3[1])
    return [location1, location2]                

paramsf = {}
continuousf = False
spacef = [[2,0],[0,2]]
F = DeterNode('F0', Fadjust, paramsf, continuousf, space=spacef,\
               description='', \
               basename='F', time=0)

CPT1 = np.array([.1, .1, .1, .1, .6])
par1 = []
space1 = [up, down, left, right, stay]
CPTip1 = (CPTip1, par1, space1)
C1 = ChanceNode('C10', distip=distip1, description='', basename='C1', time=0)

CPT2 = np.array([.1, .1, .1, .1, .6])
par2 = []
space2 = [up, down, left, right, stay]
CPTip2 = (CPT2, par2, space2)
C2 = ChanceNode('C20', distip=CPTip2, description='', basename='C2', time=0)

def adjust1(var1, var2):
    opponent = adjust(var1+var2[1]) 
    return [opponent, var2[0]]                   

paramsseek = {var1: C1, var2: F}
continuousseek = False
spaceseek = [[(w,x), (y,z)] for w in range(3) for x in range(3) \
            for y in range(3) for z in range(3)]
Fseek = DeterNode('Fseek0', adjust1, paramsseek, continuousseek, \
              space=spaceseek, basename='Fseek', time=0)

def adjust2(var1, var2):
    opponent = adjust(var1+var2[0])
    return [opponent, var2[1]] 
    
paramshide = {var1: C2, var2: F}
Fhide = DeterNode('Fhide0', adjust2, paramshide, continuousseek,
              space=spaceseek, basename='Fhide', time=0)

D1 = DecisionNode('D10', 'seeker', [up, down, left, right, stay], \
                    parents=[Fseek], basename='D1', time=0)
                    
D2 = DecisionNode('D20', 'hider', [up, down, left, right, stay], \
                    parents=[Fhide], basename='D2', time=0)

nodeset = set([F,Fseek,Fhide,C1,C2,D1,D2])
               
for t in range(1,10):
    
    paramsf = {var1: D1, var2: D2, var3: F}
    Fa = DeterNode('F%s' %t, Fadjust, paramsf, continuousf, space=spaceseek,\
                   basename='F', time=t)
    F = Fa
    nodeset.add(F)
                    
    C1 = ChanceNode('C1%s' %t, CPTip=CPTip1, basename='C1', time=t)
    nodeset.add(C1)
    
    C2 = ChanceNode('C2%s' %t, CPTip=CPTip2, basename='C2', time=t)
    nodeset.add(C2)
                    
    paramsseek = {var1: C1, var2: F}
    Fseek = DeterNode('Fseek%s' %t, adjust1, paramsseek, continuousseek, \
              space=spaceseek, basename='Fseek', time=t)
    nodeset.add(Fseek)
    
    paramshide = {var1: C2, var2: F}
    Fhide = DeterNode('Fhide%s' %t, adjust2, paramshide, continuousseek, \
              space=spaceseek, basename='Fhide', time=t)
    nodeset.add(Fhide)
              
    D1 = DecisionNode('D1%s' %t, 'seeker', [up, down, left, right, stay], \
                    parents=[Fseek], basename='D1', time=t)
    nodeset.add(D1)
                    
    D2 = DecisionNode('D2%s' %t, 'hider', [up, down, left, right, stay], \
                    parents=[Fhide], basename='D2', time=t)
    nodeset.add(D2)
    
def reward1(F=[[2,0],[0,2]]):
    if np.array_equal(F[0], F[1]):
        return 1
    else:
        return 0
    
def reward2(F=[[2,0],[0,2]]):
    return -1*reward1(F)

rfuncs = {'seeker': reward1, 'hider': reward2}
G = iterSemiNFG(nodeset, rfuncs)

for agent in G.players:
    for n in G.partition[agent]:
        n.uniformCPT()
#        
G1 = ewma_jaakkola(G, 'D2', 10, 100, .75, .8, 0.1)
#G.reward('1', 2)
#G.draw_graph()
#runfile(r'/Users/jamesbono/Documents/libnfg/PyNFG/bin/PGT_intelligence_test_script.py', \
#        wdir=r'/Users/jamesbono/Documents/libnfg/PyNFG/bin')
#from PGT_intelligence_test_script import intelligence
#intel = intelligence(G, S=5, M=20)