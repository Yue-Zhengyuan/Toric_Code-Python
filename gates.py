# 
#   gates.py
#   Toric_Code-Python
#   generate time-evolution gates and swap gates
#
#   created on Jan 21, 2019 by Yue Zhengyuan
#   code influcenced by finite_TMPS - trotter.h
#                   and iTensor - bondgate.h
#

import numpy as np
import sys
from itertools import product
import para_dict as p
import copy

# index order convention
# 
# Tensor
#       a      c
#      _|_    _|_
#  i --| |----| |--j
#      -|-    -|-
#       b      d
#
#  index order: iabcdj
# 
# Gate
#       a      c
#      _|_    _|_
#      | |----| |
#      -|-    -|-
#       b      d
#
#  index order: abcd

# exp(x) = 1 + x +  x^2/2! + x^3/3! ..
# = 1 + x * (1 + x/2 *(1 + x/3 * (...
# ~ ((x/3 + 1) * x/2 + 1) * x + 1

# 4-site time-evolution gate
def toExpH(ham, order):
    term = copy.copy(ham)
    unit = np.einsum('ab,cd,ef,gh->abcdefgh', p.iden, p.iden, p.iden, p.iden)
    for i in np.arange(order, 0, -1, dtype=int):
        term /= i
        gate = unit + term
        term = np.einsum('abcdefgh,bwdxfyhz->awcxeygz', gate, ham)
    return gate

class gate(object):
    # input parameters
    # sites: involved sites
    # putsite: marking sites already applied to magnetic field
    # kind: tEvolP (plaquette with field) / tEvolV (vertex) / Swap
    # para: parameter dictionary
    def __init__(self, sites, putsite, kind, para):
        # members of Gate
        self.sites = sites.copy()
        self.sites.sort()
        self.kind = kind
        siteNum = len(self.sites)
        expOrder = 100
        # create swap gate
        if ((self.kind == 'Swap') and (siteNum == 2)):
            self.gate = np.zeros((2,2,2,2), dtype=complex)
            for a,b,c,d in product(range(2), repeat=4):
                if (a == d and b == c):
                    self.gate[a,b,c,d] = 1.0
        # first create local Hamiltonian
        # then convert to time evolution gate via toExpH

        # Plaquette operator (with field)
        elif ((self.kind == 'tEvolP') and (siteNum == 4)):
            # plaquette part
            ham = np.einsum('ab,cd,ef,gh->abcdefgh', p.sx, p.sx, p.sx, p.sx) * (-para['p_g'])
            # field part
            if (not putsite[self.sites[0]]):
                ham += np.einsum('ab,cd,ef,gh->abcdefgh', p.sz, p.iden, p.iden, p.iden) * (-para['hz'])
            if (not putsite[self.sites[1]]):
                ham += np.einsum('ab,cd,ef,gh->abcdefgh', p.iden, p.sz, p.iden, p.iden) * (-para['hz'])
            if (not putsite[self.sites[2]]):
                ham += np.einsum('ab,cd,ef,gh->abcdefgh', p.iden, p.iden, p.sz, p.iden) * (-para['hz'])
            if (not putsite[self.sites[3]]):
                ham += np.einsum('ab,cd,ef,gh->abcdefgh', p.iden, p.iden, p.iden, p.sz) * (-para['hz'])
            ham *= (-para['tau'] / 2) * 1.0j
            self.gate = toExpH(ham, expOrder)
        # Vertex operator
        elif ((self.kind == 'tEvolV') and (siteNum == 4)):
            ham = np.einsum('ab,cd,ef,gh->abcdefgh', p.sz, p.sz, p.sz, p.sz) * (-para['v_U'])
            ham *= (-para['tau'] / 2) * 1.0j
            self.gate = toExpH(ham, expOrder)
        # Error handling
        else:
            print('Wrong parameter for gate construction.\n')
            sys.exit()

# clean redundant swap gates
def cleanGates(gateList):
    j = 0
    while j < len(gateList) - 1:
        if (gateList[j].kind == 'Swap' and gateList[j + 1].kind == 'Swap'
        and gateList[j].sites[0] == gateList[j + 1].sites[0]
        and gateList[j].sites[1] == gateList[j + 1].sites[1]):
            del gateList[j]
        else:
            j = j + 1


def makeGateList(allsites, para):
    gateList = []
    siteNum = len(allsites)
    putsite = [False] * siteNum
    swapGates = []
    # open boundary condition
    # make plaquette gates (together with field)
    if (para['p_g'] != 0):
        for i in np.arange(1, p.n - (para['nx']-1), 2 * para['nx'] - 1, dtype=int):
            for j in np.arange(0, para['nx'] - 1, 1, dtype=int):
                u = i + j
                l = u + para['nx'] - 1
                r = l + 1
                d = l + para['nx']
                sites = [u - 1, l - 1, r - 1, d - 1]
                sites.sort()
                # create swap gates
                for site in np.arange(sites[0], sites[1]-1, 1, dtype=int):
                    swapGates.append(gate([site, site + 1], putsite, 'Swap', para))
                for site in np.arange(sites[3]-1, sites[2], -1, dtype=int):
                    swapGates.append(gate([site, site + 1], putsite, 'Swap', para))
                gateSites = [sites[1]-1, sites[1], sites[1]+1, sites[1]+2]
                for k in range(len(swapGates)):
                    gateList.append(swapGates[k])
                # evolution gate (plaquette + field)
                gateList.append(gate(gateSites, putsite, 'tEvolP', para))
                # put sites back to the original place
                for k in reversed(range(len(swapGates))):
                    gateList.append(swapGates[k])
                swapGates.clear()

    # make vertex gates
    if (para['v_U'] != 0):
        for i in np.arange(para['nx'] + 1, p.n - 3 * para['nx'] + 2, 2 * para['nx'] - 1, dtype=int):
            for j in np.arange(0, para['nx'] - 2, 1, dtype=int):
                u = i + j
                l = u + para['nx'] - 1
                r = l + 1
                d = l + para['nx']
                sites = [u - 1, l - 1, r - 1, d - 1]
                sites.sort()
                # create swap gates
                for site in np.arange(sites[0], sites[1]-1, 1, dtype=int):
                    swapGates.append(gate([site, site + 1], putsite, 'Swap', para))
                for site in np.arange(sites[3]-1, sites[2], -1, dtype=int):
                    swapGates.append(gate([site, site + 1], putsite, 'Swap', para))
                gateSites = [sites[1]-1, sites[1], sites[1]+1, sites[1]+2]
                for k in range(len(swapGates)):
                    gateList.append(swapGates[k])
                # evolution gate
                gateList.append(gate(gateSites, putsite, 'tEvolV', para))
                # put sites back to the original place
                for k in reversed(range(len(swapGates))):
                    gateList.append(swapGates[k])
                swapGates.clear()

    # second order Trotter decomposition
    # b1.b2.b3....b3.b2.b1
    # append "reversed gateList" to "gateList"
    gateNum = len(gateList)
    for i in reversed(range(gateNum)):
        gateList.append(gateList[i])
    
    cleanGates(gateList)
    return gateList