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

# exp(x) = 1 + x +  x^2/2! + x^3/3! ..
# = 1 + x * (1 + x/2 *(1 + x/3 * (...
# ~ ((x/3 + 1) * x/2 + 1) * x + 1
def toExpH(ham, order):
    legnum = len(np.shape(ham))
    reshaped_ham = np.reshape(ham, (2**int(legnum/2), 2**int(legnum/2)))
    term = reshaped_ham.copy()
    unit = np.eye(2**int(legnum/2), dtype=complex)
    for i in np.arange(order, 0, -1, dtype=int):
        term /= i
        gate = unit + term
        term = np.dot(gate, reshaped_ham)
    newshape = []
    for i in range(legnum):
        newshape.append(2)
    gate = np.reshape(gate, tuple(newshape))
    return gate

class gate(object):
    # input parameters
    # sites: involved sites
    # kind: tEvolP (plaquette) / tEvolV (vertex) / tEvolF (field) / Swap
    # para: parameter dictionary
    def __init__(self, sites, kind, para):
        # members of Gate
        self.sites = sites.copy()
        self.kind = kind

        # create swap gate
        if ((self.kind == 'Swap') and (len(self.sites) == 2)):
            self.gate = np.einsum('ab,cd->acbd', p.iden, p.iden)
        # first create local Hamiltonian
        # then convert to time evolution gate via toExpH
        elif ((self.kind == 'tEvolP') and (len(self.sites) == 4)):
            ham = np.einsum('ab,cd,ef,gh->acegbdfh', p.sx, p.sx, p.sx, p.sx) * (-para['p_g'])
            self.gate = toExpH(ham, 100)
        elif ((self.kind == 'tEvolV') and (len(self.sites) == 4)):
            ham = np.einsum('ab,cd,ef,gh->acegbdfh', p.sz, p.sz, p.sz, p.sz) * (-para['v_U'])
            self.gate = toExpH(ham, 100)
        elif ((self.kind == 'tEvolF') and (len(self.sites) == 1)):
            ham = p.sz * (-para['hz'])
            self.gate = toExpH(ham, 100)
        else:
            print('Wrong parameter for gate construction.\n')
            sys.exit()


def makeGateList(sites, para):
    gateList = []

    # make field gates (no need for swap gates)
    if (para['hz'] != 0):
        for i in range(p.n):
            current_site = [i]
            gateList.append(gate(current_site, 'tEvolF', para))
    # make vertex gates
    if (para['v_U'] != 0):
        for i in np.arange(para['nx'] + 1, p.n - para['nx'] + 2, 2 * para['nx'] - 1, dtype=int):
            for j in np.arange(0, para['nx'] - 2, 1, dtype=int):
                u = i + j
                l = u + para['nx'] - 1
                r = l + 1
                d = l + para['nx']
                if (d > p.n):
                    d = d - p.n
                # move u to l - 1
                for site in np.arange(u, l, 1, dtype=int):
                    gateList.append(gate([site, site + 1], 'Swap', para))
                # evolution gate

                # move d to r + 1
                for site in np.arange(d, r, -1, dtype=int):
                    gateList.append(gate([site - 1, site], 'Swap', para))

    # make plaquette gates
    if (para['p_g'] != 0):
        for i in np.arange(1, p.n, 2 * para['nx'] - 1, dtype=int):
            for j in np.arange(0, para['nx'] - 1, 1, dtype=int):
                u = i + j
                l = u + para['nx'] - 1
                r = l + 1
                d = l + para['nx']
                if (d > p.n):
                    d = d - p.n
                # move u to l - 1
                for site in np.arange(u, l, 1, dtype=int):
                    gateList.append(gate([site, site + 1], 'Swap', para))
                # evolution gate

                # move d to r + 1
                for site in np.arange(d, r, -1, dtype=int):
                    gateList.append(gate([site - 1, site], 'Swap', para))
    return gateList