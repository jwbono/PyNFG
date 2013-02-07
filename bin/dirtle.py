# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:13:34 2012

@author: jamesbono
"""

class Base(object):
    def __init__(self, name):
        self.name = name
        
    def sayname(self):
        print 'Base: ',self.name
        
class Derived(Base):
    def __init__(self, name, doggy):
        self.name = name
        self.doggy = doggy
        
    def saydog(self):
        print 'Derived: ',self.doggy
        
B = Base('Dad')
B.sayname()
D = Derived('Son', 'Jackson')
D.sayname()
D.saydog()
#def deterfunc(x=1, y=2, z=3):
#    r=x+y+z
#    return r
#
#r = deterfunc(**{'x': 3, 'z':5})
#print r
#import pymc as pm
#
#class StcChanceNode(object):
#    def __init__(self, name, space, funname, args, kwargs):
#        z = lambda x, y: eval(funname)(*x, **y)
#        self.pymc = z(args, kwargs)
#        for keys in self.pymc.parents:
#            self.parents = eval
#
#name = 'Yildiray'
#funname = 'pm.DiscreteUniform'
#space = range(6)
#args = [name, 0, 5]
#kwargs = {}
#kwargs['value'] = 2
#
#NFG_node = StcChanceNode(name, space, funname, args, kwargs)

#from time import clock, time
#from scipy.stats import distributions as rv
#
#a = 30
#b = 11
#c = 5
#d = 15
#
#y = rv.randint.rvs(0, c, size=(d,a,b))
#x = np.zeros((d,a,b,c))
#
#start = time()
#shape_last = x.shape[-1]
#x.reshape((-1, shape_last))[np.arange(y.size), y.flatten()]=1
#elapsed = (time()-start)
#print(elapsed)

#start = time()
#for i in range(2):
#    for j in range(3):
#        x[i,j,y[i,j]]=1
#elapsed = (time()-start)
#print(elapsed)