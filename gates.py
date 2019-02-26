# 
#   gates.py
#   Toric_Code-Python
#   generate time-evolution gates and swap gates
#   field are added with plaquettes
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

# exp(x) = 1 + x +  x^2/2! + x^3/3! ..
# = 1 + x * (1 + x/2 *(1 + x/3 * (...
# ~ ((x/3 + 1) * x/2 + 1) * x + 1

def toExpH4(ham, order):
    """
    Create 4-site time-evolution gate using the approximation

        exp(x) = 1 + x + x^2/2! + x^3/3! ..
        = 1 + x * (1 + x/2 *(1 + x/3 * (...
        ~ ((x/3 + 1) * x/2 + 1) * x + 1

    Parameters
    ---------------
    ham : list of length 4 of numpy arrays
        4-site local Hamiltonian
    order : int
        approximation order in the Taylor series
    """
    term = copy.copy(ham)
    unit = np.einsum('ab,cd,ef,gh->abcdefgh', p.iden, p.iden, p.iden, p.iden)
    for i in np.arange(order, 0, -1, dtype=int):
        term /= i
        gate = unit + term
        term = np.einsum('abcdefgh,bwdxfyhz->awcxeygz', gate, ham)
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
    kind : 'tEvolP'/'tEvolV'/'Swap'
        kind of the gate
    args : dictionary
        parameters of the system
    """
    def __init__(self, sites, putsite, kind, args):
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
        #   H = - U * \sum(A_p) - g * \sum(B_p) - hz * \sum(Sz) - hx * \sum(Sx)
        # Plaquette operator (with field)
        elif ((self.kind == 'tEvolP') and (siteNum == 4)):
            ham = np.einsum('ab,cd,ef,gh->abcdefgh', p.sx, p.sx, p.sx, p.sx) * (-args['g'])
            # adding field
            if (args['hx'] != 0 or args['hy'] != 0 or args['hz'] != 0):
                if args['hx'] != 0:
                    field = args['hx']
                    pauli = p.sx
                elif args['hy'] != 0:
                    field = args['hy']
                    pauli = p.sy
                elif args['hz'] != 0:
                    field = args['hz']
                    pauli = p.sz
                # else: 
                #     sys.exit('Added field in more than one direction')
                for i in range(4):
                    if (not putsite[sites[i]]):
                        mat_list = [p.iden] * 4
                        mat_list[i] = pauli
                        ham += np.einsum('ab,cd,ef,gh->abcdefgh', mat_list[0], mat_list[1], mat_list[2], mat_list[3]) * (-field)
            # create gate exp(-iHt/2)
            ham *= (-args['tau'] / 2) * 1.0j
            self.gate = toExpH4(ham, expOrder)

        # Vertex operator
        elif ((self.kind == 'tEvolV') and (siteNum == 4)):
            ham = np.einsum('ab,cd,ef,gh->abcdefgh', p.sz, p.sz, p.sz, p.sz) * (-args['U'])
            ham *= (-args['tau'] / 2) * 1.0j
            self.gate = toExpH4(ham, expOrder)
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
            del gateList[j + 1]
            del gateList[j]
            j -= 1
        else:
            j += 1

def makeGateList(allsites, args):
    """
    Create time-evolution/swap gate list

    Parameters
    ---------------
    allsites : list of numpy arrays
        MPS/MPO to be acted on
    args : dictionary
        parameter dictionary
    """
    gateList = []
    siteNum = len(allsites)
    putsite = [False] * siteNum
    swapGates = []
    # open boundary condition
    # make plaquette gates (together with field)
    if (args['g'] != 0):
        for i in np.arange(1, p.n - (args['nx']-1), 2 * args['nx'] - 1, dtype=int):
            for j in np.arange(0, args['nx'] - 1, 1, dtype=int):
                u = i + j
                l = u + args['nx'] - 1
                r = l + 1
                d = l + args['nx']
                sites = [u - 1, l - 1, r - 1, d - 1]
                sites.sort()
                # create swap gates
                for site in np.arange(sites[0], sites[1]-1, 1, dtype=int):
                    swapGates.append(gate([site, site + 1], putsite, 'Swap', args))
                for site in np.arange(sites[3]-1, sites[2], -1, dtype=int):
                    swapGates.append(gate([site, site + 1], putsite, 'Swap', args))
                gateSites = [sites[1]-1, sites[1], sites[1]+1, sites[1]+2]
                for k in range(len(swapGates)):
                    gateList.append(swapGates[k])
                # evolution gate (plaquette with field)
                gateList.append(gate(gateSites, putsite, 'tEvolP', args))
                """
                VERY IMPORTANT:
                the field is added AFTER swap gate has been applied

                Example
                -----------
                Suppose a 4-site operator acts on [11,15,16,20]
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

    # make vertex gates
    if (args['U'] != 0):
        for i in np.arange(args['nx'] + 1, p.n - 3 * args['nx'] + 2, 2 * args['nx'] - 1, dtype=int):
            for j in np.arange(0, args['nx'] - 2, 1, dtype=int):
                u = i + j
                l = u + args['nx'] - 1
                r = l + 1
                d = l + args['nx']
                sites = [u - 1, l - 1, r - 1, d - 1]
                sites.sort()
                # create swap gates
                for site in np.arange(sites[0], sites[1]-1, 1, dtype=int):
                    swapGates.append(gate([site, site + 1], putsite, 'Swap', args))
                for site in np.arange(sites[3]-1, sites[2], -1, dtype=int):
                    swapGates.append(gate([site, site + 1], putsite, 'Swap', args))
                gateSites = [sites[1]-1, sites[1], sites[1]+1, sites[1]+2]
                for k in range(len(swapGates)):
                    gateList.append(swapGates[k])
                # evolution gate (vertex)
                gateList.append(gate(gateSites, putsite, 'tEvolV', args))
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