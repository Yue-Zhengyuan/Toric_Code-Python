# 
#   expHam.py
#   Toric_Code-Python
#   Create MPO form of Toric Code H and exp(-iH dt)
#
#   created on Mar 28, 2019 by Yue Zhengyuan
#

import numpy as np
import sys
import copy
import mps
import mpo
import para_dict as p
from itertools import product


def Ham_builder(args):
    nx, ny, U, g, hz = args['nx'],args['ny'],args['U'], args['g'], args['hz']
    ham = []
    iden = [np.reshape(p.iden, (1,2,2,1))] * args['real_n']
    yperiodic = args['yperiodic']
    # plaquette (XXXX)
    if (args['g'] != 0):
        # i -> row; j -> column
        for i in np.arange(1, args['n'] - (args['nx']-1), 2 * args['nx'] - 1, dtype=int):
            for j in np.arange(0, args['nx'] - 1, 1, dtype=int):
                u = i + j
                l = u + args['nx'] - 1
                r = l + 1
                d = l + args['nx']
                sites = [u - 1, l - 1, r - 1, d - 1]
                if yperiodic == True:
                    sites[3] -= args['real_n']
                sites.sort()
                term = copy.copy(iden)
                for site in sites:
                    term[site] = np.reshape(p.sx * (-g), (1,2,2,1))
                if ham == []: # initialize
                    ham = copy.copy(term)
                else:
                    ham = mpo.sum(ham, term, args, compress=None)

    # vertex (ZZZZ)
    if (args['U'] != 0):
        for i in np.arange(args['nx'] + 1, args['n'] - 3 * args['nx'] + 2, 2 * args['nx'] - 1, dtype=int):
            for j in np.arange(0, args['nx'] - 2, 1, dtype=int):
                u = i + j
                l = u + args['nx'] - 1
                r = l + 1
                d = l + args['nx']
                sites = [u - 1, l - 1, r - 1, d - 1]
                # inRegion = True
                # for site in sites:
                #     if site in region:
                #         pass
                #     else:
                #         inRegion = False
                #         continue
                sites.sort()
                term = copy.copy(iden)
                for site in sites:
                    term[site] = np.reshape(p.sz * (-U), (1,2,2,1))
                if ham == []: # initialize
                    ham = copy.copy(term)
                else:
                    ham = mpo.sum(ham, term, args, compress=None)
    # field
    for i in range(args['real_n']):
        term = copy.copy(iden)
        term[i] = np.reshape(p.sz * (-hz), (1,2,2,1))
        if ham == []: # initialize
            ham = copy.copy(term)
        else:
            ham = mpo.sum(ham, term, args, compress=None)
    
    # svd compression
    ham = mpo.position(ham, 0, args)
    ham = mpo.position(ham, len(ham)-1, args, oldcenter=0)
    return ham

def exp_Ham(ham, dt):
    # exp(iHt) = 1 + (iHt) + (iHt)^2/2! + ...
    expham = [np.reshape(p.iden, (1,2,2,1))] * len(ham)
    return expham