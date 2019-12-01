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
import lattice as lat

# exp(x) = 1 + x +  x^2/2! + x^3/3! ..
# = 1 + x * (1 + x/2 *(1 + x/3 * (...
# ~ ((x/3 + 1) * x/2 + 1) * x + 1

def toExpH(ham, order):
    """
    Create 2/3/4-site time-evolution gate using the approximation

        exp(x) = 1 + x + x^2/2! + x^3/3! ..
        = 1 + x * (1 + x/2 *(1 + x/3 * (...
        ~ ((x/3 + 1) * x/2 + 1) * x + 1

    Parameters
    ---------------
    ham : numpy array
        2/3/4-site local Hamiltonian
    order : int
        approximation order in the Taylor series
    """
    siteNum = int(len(np.shape(ham)) / 2)
    if (siteNum != 2 and siteNum != 3 and siteNum != 4):
        sys.exit("Wrong number of sites")

    term = copy.copy(ham)
    mat_list = [p.iden] * siteNum
    if siteNum == 2:
        unit = np.einsum('ab,cd->abcd', *mat_list)
    elif siteNum == 3:
        unit = np.einsum('ab,cd,ef->abcdef', *mat_list)
    elif siteNum == 4:
        unit = np.einsum('ab,cd,ef,gh->abcdefgh', *mat_list)
    
    for i in np.arange(order, 0, -1, dtype=int):
        term /= i
        gate = unit + term
        if siteNum == 2:
            term = np.einsum('abcd,bwdx->awcx', gate, ham)
        elif siteNum == 3:
            term = np.einsum('abcdef,bwdxfy->awcxey', gate, ham)
        elif siteNum == 4:
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
        # the sites are ordered to take PBC into account
        # self.sites.sort()
        self.kind = kind
        siteNum = len(self.sites)
        expOrder = 50
        # create swap gate
        if ((self.kind == 'Swap') and (siteNum == 2)):
            self.gate = np.zeros((2,2,2,2), dtype=complex)
            for a,b,c,d in product(range(2), repeat=4):
                if (a == d and b == c):
                    self.gate[a,b,c,d] = 1.0

        # first create local Hamiltonian
        # then convert to time evolution gate via toExpH

        # System Hamiltonian:
        #   H = - U * \sum(A_p) - g * \sum(B_p) - hz * \sum(Sz)
        # Plaquette operator (with field)
        elif (self.kind == 'tEvolP'):
            mat_list = [p.sx] * siteNum
            ham = copy.copy(mat_list[0])
            for i in range(1, siteNum):
                ham = np.tensordot(ham, mat_list[i], axes=0)
            ham *= (-args['g'])
            # adding field
            if (args['hz'] != 0):
                for i in range(siteNum):
                    if (not putsite[sites[i]]):
                        mat_list = [p.iden] * siteNum
                        mat_list[i] = p.sz
                        addterm = copy.copy(mat_list[0])
                        for j in range(1, siteNum):
                            addterm = np.tensordot(addterm, mat_list[j], axes=0)
                        addterm *= (-args['hz'])
                        ham += addterm
            # create gate exp(-iHt/2)
            ham *= (-args['tau'] / 2) * 1.0j
            self.gate = toExpH(ham, expOrder)

        # Vertex operator
        elif (self.kind == 'tEvolV'):
            mat_list = [p.sz] * siteNum
            ham = copy.copy(mat_list[0])
            for i in range(1, siteNum):
                ham = np.tensordot(ham, mat_list[i], axes=0)
            # ham = np.einsum('ab,cd,ef,gh->abcdefgh', *mat_list) * (-args['U'])
            ham *= (-args['U'])
            ham *= (-args['tau'] / 2) * 1.0j
            self.gate = toExpH(ham, expOrder)
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

def makeGateList(siteNum, args, region=range(10000)):
    """
    Create time-evolution/swap gate list

    Parameters
    ---------------
    siteNum : int
        number of sites
    args : dictionary
        parameter dictionary
    region : iterable of integers (default range(p.args['n']))
        the region from which gates are constructed
    """
    gateList = []
    xperiodic = args['xperiodic']
    putsite = [False] * siteNum
    swapGates = []
    # open boundary condition along x
    # make plaquette gates (together with field)
    if (args['g'] != 0):
        # i -> row; j -> column
        for j, i in product(range(args['ny'] - 1), range(args['nx'] - 1)):
            out_of_region = False
            u = lat.lat((  i,   j), 'r', (args['nx'], args['ny']), xperiodic=xperiodic)
            l = lat.lat((  i,   j), 'd', (args['nx'], args['ny']), xperiodic=xperiodic)
            r = lat.lat((i+1,   j), 'd', (args['nx'], args['ny']), xperiodic=xperiodic)
            d = lat.lat((  i, j+1), 'r', (args['nx'], args['ny']), xperiodic=xperiodic)
            sites = [u, l, r, d]
            # check if this gate is in the region
            for i in range(len(sites)):
                if sites[i] not in region:
                    out_of_region = True
                    break
            if out_of_region == True:
                continue
            # print(sites)
            # create swap gates
            # reaching boundary
            if (r % (args['nx'] - 1) == 0 and xperiodic == True):
                # in this case r = u + 1, l = u + nx - 1, d = u + 2 (nx - 1)
                sites.sort()
                # now sites = [u, u + 1, u + nx - 1, u + 2 (nx - 1)]
                # move sites[2]
                for site in np.arange(sites[2], sites[1] + 1, -1, dtype=int):
                    swapGates.append(gate([site - 1, site], putsite, 'Swap', args))
                # move sites[3]
                for site in np.arange(sites[3], sites[1] + 2, -1, dtype=int):
                    swapGates.append(gate([site - 1, site], putsite, 'Swap', args))
                gateSites = [sites[0], sites[0]+1, sites[0]+2, sites[0]+3]
            else:
                # OBC case / PBC case off boundary
                # move sites[0]
                for site in np.arange(sites[0], sites[1] - 1, 1, dtype=int):
                    swapGates.append(gate([site, site + 1], putsite, 'Swap', args))
                # move sites[3]
                for site in np.arange(sites[3], sites[2] + 1, -1, dtype=int):
                    swapGates.append(gate([site - 1, site], putsite, 'Swap', args))
                gateSites = [sites[1]-1, sites[1], sites[1]+1, sites[1]+2]

            for k in range(len(swapGates)):
                gateList.append(swapGates[k])
            # evolution gate (plaquette with field)
            gateList.append(gate(gateSites, putsite, 'tEvolP', args))
            """
            VERY IMPORTANT:
            how to add magnetic field to the sites

            Example
            -----------
            Suppose a 4-site operator acts on [11,15,16,20]
            It will become [14,15,16,17] after applying the swap gates 
            We NOW add field to site 14,15,16,17 instead of 11,15,16,20
            But we will set putsite[11,15,16,20] (not [14,15,16,17]) to True
            """
            for site in sites:
                putsite[site] = True
            # put sites back to the original place
            for k in reversed(range(len(swapGates))):
                gateList.append(swapGates[k])
            swapGates.clear()
    # vertex gates (ZZZZ) are not necessary since it commutes with 
    # plaquette (XXXX), closed string (X...X) and field (Z)
    # make vertex gates
    if (args['U'] != 0):
        # i -> row; j -> column
        if xperiodic == True:
            irange = range(args['nx'] - 1)
        else:
            irange = range(args['nx'])
        for j, i in product(range(args['ny'] - 1), irange):
            out_of_region = False
            sites = lat.vertex((i, j), args)
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