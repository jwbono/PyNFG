import numpy as np


### See the notebook for the scenario.  Run this script to run
### the actual test.

#######################################################
### Computing the analytical equilibrium strategies ###
#######################################################

probs = np.ones(6)/float(6)
strats = np.arange(10,70,10)

def EU_H(Q, probsopp =probs):
    eus = []
    for i in Q:
        u = probsopp*(i*(200-2*i-2*strats) - i)
        eus.append(np.sum(u))
    return np.asarray(eus)

def EU_L(Q, probsopp =probs):
    eus = []
    for i in Q:
        u = probsopp* (i*(90-i-strats)-i)
        eus.append(np.sum(u))
    return np.asarray(eus)

def lqre(eus, beta):
    denom = np.sum(np.exp(beta*eus))
    num = np.exp(beta*eus)
    return num/denom

alpha = .8 #high_given_high
gamma = .3 #high_given_low

def phigh(signal):
    if signal =='h':
        return alpha*.5/(alpha*.5 +gamma*.5)
    if signal == 'l':
        return .5*(1-alpha)/(.5*(1-alpha) + .5 * (1-gamma))


def plow(signal):
    if signal =='h':
        return gamma*.5/(gamma*.5 + alpha*.5)
    if signal =='l':
        return .5 *(1-gamma) / (.5*(1-gamma) + .5*(1-alpha))

def treylk(signal, mikeshigh=probs, mikeslow=probs,
           beta=.01, returnEU=False):
    if signal == 'h':
        EU = (alpha *EU_H(strats, mikeshigh) +
              (1-alpha)*EU_H(strats, mikeslow))
        if returnEU:
            return EU
        return lqre(EU, beta)
    if signal == 'l':
        EU = (gamma *EU_L(strats, mikeshigh) +
              (1-gamma)*EU_L(strats, mikeslow))
        if returnEU:
            return EU
        return lqre(EU, beta)

def mikelk(signal, treyhigh = probs, treylow = probs,
           beta=.01, returnEU=False):
    if signal =='h':
        high_m = phigh('h')
        EU = (high_m * EU_H(strats, treyhigh) +
              (1-high_m)*EU_L(strats, treylow))
        if returnEU:
            return EU
        return lqre(EU, beta)
    if signal =='l':
        high_m = phigh('l')
        EU = (high_m * EU_H(strats, treyhigh) +
              (1-high_m)*EU_L(strats, treylow))
        if returnEU:
            return EU
        return lqre(EU, beta)

def genCPT(Lmike=3, Ltrey=3):
    mikelkh = np.copy(probs)
    mikelkl = np.copy(probs)
    treylkh = np.copy(probs)
    treylkl = np.copy(probs)
    CPT ={'trey': None, 'mike': None}
    if Lmike == 0:
        CPT['mike'] = np.array([mikelkh, mikelkl])
    if Ltrey == 0:
        CPT['trey'] = np.array([treylkh, treylkl])
    for lev in range(max(Lmike, Ltrey)):
        mikelkh_temp, mikelkl_temp = \
          mikelk('h', treyhigh = treylkh, treylow = treylkl), \
          mikelk('l', treyhigh = treylkh, treylow = treylkl)
        treylkh, treylkl = \
          treylk('h', mikeshigh=mikelkh, mikeslow=mikelkl),\
          treylk('l', mikeshigh=mikelkh, mikeslow=mikelkl)
        mikelkh, mikelkl = mikelkh_temp, mikelkl_temp
        if Lmike ==lev +1:
            CPT['mike'] = np.array([mikelkh, mikelkl])
        if Ltrey ==lev+1:
            CPT['trey'] = np.array([treylkh, treylkl])
    return CPT

# The level 2 Logit QRE equilibrium strategy
CPTs = genCPT(2,2)

import pynfg as pynfg
market = pynfg.ChanceNode('market', (np.array([.5,.5]), [], ['h', 'l']))
trey = pynfg.DecisionNode('trey', 'trey', list(np.arange(10,70,10)),
                          parents =[market])
mikes_signal = pynfg.ChanceNode('mikes_signal',
                                (np.array([[.8,.2], [.3,.7]]),
                                 [market], ['hi', 'l0']))
mike = pynfg.DecisionNode('mike', 'mike', list(np.arange(10,70,10)),
                          parents = [mikes_signal])

def umike(market, trey, mike):
    if market =='h':
        return mike*(200-2*(mike+trey)) - mike
    if market =='l':
        return mike * (90-(mike+trey)) -mike

def utrey(market, trey, mike):
    if market =='h':
        return trey*(200-2*(mike+trey)) - trey
    if market =='l':
        return trey * (90-(mike+trey)) -trey
utils = {'mike': umike, 'trey': utrey}

Game =pynfg.SemiNFG(set([market, trey, mikes_signal, mike]), utils)


params = pynfg.levelksolutions.br_dict(Game, 40000, 500,
                                tol=10000, L0Dist='uniform', beta=.01)
brgame = pynfg.levelksolutions.BestResponse(Game, params)
brgame.Game.node_dict['market'].set_value('h')
brgame.Game.node_dict['market'].draw_value()
brgame.Game.node_dict['mikes_signal'].draw_value()
brgame.train_node('trey', 1, logit=True)
brgame.train_node('mike', 1, logit=True)
brgame.train_node('trey', 2, logit=True)
brgame.train_node('mike', 2, logit=True)
from numpy.testing import assert_almost_equal
assert_almost_equal(brgame.Game.node_dict['trey'].LevelCPT[2],
                    CPTs['trey'], decimal=2)
assert_almost_equal(brgame.Game.node_dict['mike'].LevelCPT[2],
                    CPTs['mike'], decimal=2)
