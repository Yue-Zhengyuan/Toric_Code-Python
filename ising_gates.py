# 
#   ising_gates.py
#   Toric_Code-Python
#   generate time-evolution gates and swap gates for 2D Ising model
#
#   created on Jan 21, 2019 by Yue Zhengyuan
#   code influcenced by finite_TMPS - trotter.h
#                   and iTensor - bondgate.h
#

import numpy as np
import sys
from itertools import product
import para_dict as p
import lattice
import copy

# index order convention
# 
#  Operator
#       a      c
#      _|_    _|_
#  i --| |----| |--j
#      -|-    -|-
#       b      d
#
#  index order: iabcdj
#
#  State
#       a      b
#      _|_    _|_
#  i --| |----| |--j
#      ---    ---
#
#  index order: iabj
# 
#  (2-site) Gate
#       a      c
#      _|______|_
#      |        |
#      -|------|-
#       b      d
#
#  index order: abcd

# exp(x) = 1 + x +  x^2/2! + x^3/3! ..
# = 1 + x * (1 + x/2 *(1 + x/3 * (...
# ~ ((x/3 + 1) * x/2 + 1) * x + 1

def toExpH2(ham, order):
    """
    Create 2-site time-evolution gate using the approximation

        exp(x) = 1 + x + x^2/2! + x^3/3! ..
        = 1 + x * (1 + x/2 *(1 + x/3 * (...
        ~ ((x/3 + 1) * x/2 + 1) * x + 1

    Parameters
    ---------------
    ham : list of length 2 of numpy arrays
        2-site local Hamiltonian
    order : int
        approximation order in the Taylor series
    """
    term = copy.copy(ham)
    unit = np.einsum('ab,cd->abcd', p.iden, p.iden)
    for i in np.arange(order, 0, -1, dtype=int):
        term /= i
        gate = unit + term
        term = np.einsum('abcd,bwdx->awcx', gate, ham)
    return gate

def toExpH1(ham, order):
    """
    Create 1-site time-evolution gate using the approximation

        exp(x) = 1 + x + x^2/2! + x^3/3! ..
        = 1 + x * (1 + x/2 *(1 + x/3 * (...
        ~ ((x/3 + 1) * x/2 + 1) * x + 1

    Parameters
    ---------------
    ham : list of length 1 of numpy arrays
        1-site local Hamiltonian
    order : int
        approximation order in the Taylor series
    """
    term = copy.copy(ham)
    unit = p.iden
    for i in np.arange(order, 0, -1, dtype=int):
        term /= i
        gate = unit + term
        term = np.einsum('ab,bw->aw', gate, ham)
    return gate

class gate(object):
    """
    Class of time-evolution/swap gates

    Initialization Parameters
    ----------------------------
    sites : list of integers
        number of sites on which the gate acts
    putsite : list of boolean variables
        marking whether magnetic field has been applied to a certian site;
        size of the list should equal that of the whole system
    kind : 'tEvolI'/'Swap'
        kind of the gate
    para : dictionary
        parameters of the system
    """
    def __init__(self, sites, putsite, kind, para):
        # members of Gate
        self.sites = copy.copy(sites)
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
        # System Hamiltonian:
        #   H = - J * \sum(Z_i Z_j) - hz * \sum(Sz) - hy * \sum(Sy) - hx * \sum(Sx)

        # Interaction (and Field)
        elif ((self.kind == 'tEvolI') and (siteNum == 2)):
            ham = np.einsum('ab,cd->abcd', p.sz, p.sz) * (-para['J'])
            # adding field
            if para['hx'] != 0:
                if (not putsite[sites[0]]):
                    ham += np.einsum('ab,cd->abcd', p.sx, p.iden) * (-para['hx'])
                if (not putsite[sites[1]]):
                    ham += np.einsum('ab,cd->abcd', p.iden, p.sx) * (-para['hx'])
            ham *= (-para['tau'] / 2) * 1.0j
            self.gate = toExpH2(ham, expOrder)
        
        # Error handling
        else:
            print('Wrong parameter for gate construction.\n')
            sys.exit()

def cleanGates(gateList):
    """clean redundant swap gates"""
    j = 0
    while j < len(gateList) - 1:
        if (gateList[j].kind == 'Swap' and gateList[j + 1].kind == 'Swap'
        and gateList[j].sites[0] == gateList[j + 1].sites[0]
        and gateList[j].sites[1] == gateList[j + 1].sites[1]):
            del gateList[j]
        else:
            j = j + 1

def makeGateList(allsites, para):
    """
    Create time-evolution/swap gate list

    Parameters
    ---------------
    allsites : list of numpy arrays
        MPS/MPO to be acted on
    para : dictionary
        parameter dictionary
    """
    gateList = []
    siteNum = len(allsites)
    putsite = [False] * siteNum
    swapGates = []
    # open boundary condition
    # make interaction gates (together with field)
    bondTable = lattice.lat_table(para['nx'])[0]
    for bond in bondTable:
        i, j, dir = bond[0], bond[1], bond[2]
        if dir == 'r':
            sites = [i + (j-1)*para['nx'], 1 + i + (j-1)*para['nx']]
        elif dir == 'd':
            sites = [i + (j-1)*para['nx'], i + j*para['nx']]
    
        sites.sort()
        # create swap gates
        for site in np.arange(sites[0], sites[1]-1, 1, dtype=int):
            swapGates.append(gate([site, site + 1], putsite, 'Swap', para))
        gateSites = [sites[1]-1, sites[1]]
        for k in range(len(swapGates)):
            gateList.append(swapGates[k])
        # evolution gate (interaction)
        gateList.append(gate(gateSites, putsite, 'tEvolI', para))
        """
        VERY IMPORTANT:
        the field is added AFTER swap gate has been applied

        Example
        -----------
        Suppose a plaquette acts on [11,15,16,20]
        It will become [14,15,16,17] after applying the swap gates 
        We NOW add field to site 14,15,16,17 instead of 11,15,16,20
        But we will set putsite[11,15,16,20], not [14,15,16,17] to True
        """
        for site in sites:
            putsite[site] = True
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