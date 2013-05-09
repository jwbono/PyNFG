# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:39:38 2013

Copyright (C) 2013 James Bono (jwbono@gmail.com)

GNU Affero General Public License

"""
from __future__ import division
import numpy as np

def convert_2_pureCPT(anarray):
    """Convert an arbitrary matrix to a pure CPT w/ weight on maximum elements
    
    :arg anarray: The numpy array to be converted to 
    :type anarray: np.array
    :returns: a normalized conditional probability distribution over actions
       given messages with all elements zero or one.
    
    """
    idx = np.where(anarray == np.max(anarray, axis=-1)[...,np.newaxis])
    newarray = np.zeros(anarray.shape)
    newarray[idx] = 1
    newCPT = newarray/np.sum(newarray, axis=-1)[...,np.newaxis]
    return newCPT
    
def mh_decision(pnew, pold, qnew=1, qold=1):
    """Decide to accept the new draw or keep the old one
    
    :arg pnew: the unnormalized likelihood of the new draw
    :type pnew: float
    :arg pold: the unnormalized likelihood of the old draw
    :type pnew: float
    :arg qnew: the probability of transitioning from the old draw to the new 
       draw.
    :type qnew: float
    :arg qold: the probability of transitioning from the new draw to the old 
       draw.
    :type qold: float
    :returns: either True or False to determine whether the new draw is 
       accepted.
    
    """
    if pold<=0 or qnew<=0:
        a = 1
    else:
        a = min([(pnew*qold)/(pold*qnew), 1])
    u = np.random.rand()
    if a > u:
        verdict = True
    else:
        verdict = False
    return verdict