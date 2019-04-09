# 
#   gnd_state.py
#   Toric_Code-Python
#   generate ground state of Toric Code system
#
#   created on Feb 18, 2019 by Yue Zhengyuan
#

import numpy as np
import para_dict as p
import mps
import sys
import copy
from itertools import product
from tqdm import tqdm

def gnd_state_builder(args):
    """
    Create MPS for Toric Code ground state \n
    Accepts x-PBC only
    """
    # building blocks
    # ---------------------------------------------------
    # tensors
    # virtual indices:
    # 0 -> no string; 1 -> with string
    vertex = np.zeros((2,2,2,2), dtype=complex)
    for i,j,k,l in product(range(2), repeat=4):
        indices = np.array([i,j,k,l])
        cond = np.where(indices == 0)
        zero_num = len(np.transpose(cond))
        if zero_num in [0,2,4]:
            vertex[i,j,k,l] = 1.0
    # 2nd index -> physical leg
    # 0 -> up; 1 -> down
    spin = np.zeros((2,2,2), dtype=complex)
    spin[0,0,0] = 1.0
    spin[1,1,1] = 1.0
    # virtual indices:
    # 0 -> no string; 1 -> with string
    corner = np.zeros((2,2), dtype=complex)
    corner[0,0] = 1.0
    corner[1,1] = 1.0
    # virtual indices:
    # 0 -> no string; 1 -> with string
    edge = np.zeros((2,2,2), dtype=complex)
    for i,j,k in product(range(2), repeat=3):
        indices = np.array([i,j,k])
        cond = np.where(indices == 0)
        zero_num = len(np.transpose(cond))
        if zero_num in [0,2]:
            edge[i,j,k] = 1.0
    # ---------------------------------------------------

    # construct MPS from PEPS
    nx, ny = args['nx'], args['ny']
    result = []
    i = 0
    xperiodic = args['xperiodic']

    if xperiodic == False:
        for row in range(2 * ny - 1):
            if (row == 0):
                result.append(corner)
                for j in np.arange(1, nx, 1, dtype=int):
                    result[i] = np.tensordot(result[i], spin, ([-1],[0]))
                    if j == nx - 1:
                        result[i] = np.tensordot(result[i], corner, ([-1],[0]))
                    else:
                        result[i] = np.tensordot(result[i], edge, ([-1],[0]))
                # original axes order:  vir phy vir phy ... vir phy vir
                # new axes order:       phy ... phy (vir ... vir)
                # move phy legs to front
                dest = 0
                for axis in np.arange(1, len(result[i].shape), 2, dtype=int):
                    result[i] = np.moveaxis(result[i], axis, dest)
                    dest += 1
                # reshape
                newshape = [1]
                for count in range(nx-1): 
                    newshape.append(2)
                newshape.append(2**nx)
                result[i] = np.reshape(result[i], newshape)

            elif (row == 2 * ny - 2):
                result.append(corner)
                for j in np.arange(1, nx, 1, dtype=int):
                    result[i] = np.tensordot(result[i], spin, ([-1],[0]))
                    if j == nx - 1:
                        result[i] = np.tensordot(result[i], corner, ([-1],[0]))
                    else:
                        result[i] = np.tensordot(result[i], edge, ([-1],[0]))
                # original axes order:  vir phy vir phy ... vir phy vir
                # new axes order:       vir ... vir (phy ... phy)
                # move vir legs to front
                dest = 0
                for axis in np.arange(0, len(result[i].shape), 2, dtype=int):
                    result[i] = np.moveaxis(result[i], axis, dest)
                    dest += 1
                # reshape
                newshape = [2**nx]
                for count in range(nx-1): 
                    newshape.append(2)
                newshape.append(1)
                result[i] = np.reshape(result[i], newshape)

            elif row % 2 == 0:
                result.append(edge)
                for j in np.arange(1, nx, 1, dtype=int):
                    result[i] = np.tensordot(result[i], spin, ([-1],[0]))
                    if j == nx - 1:
                        result[i] = np.tensordot(result[i], edge, ([-1],[0]))
                    else:
                        result[i] = np.tensordot(result[i], vertex, ([-1],[0]))
                # original axes order:  vir0 vir1 phy ... vir0 vir1 phy vir0 vir1
                # new axes order:       (vir0 ... vir0) phy ... phy (vir1 ... vir1)
                # move vir0 (upward) legs to front
                dest = 0
                for axis in np.arange(0, len(result[i].shape), 3, dtype=int):
                    result[i] = np.moveaxis(result[i], axis, dest)
                    dest += 1
                # current axes order: vir0 ... vir0 / vir1 phy ... vir1 phy vir1
                # move phy legs to center
                dest = nx
                for axis in np.arange(nx+1, len(result[i].shape), 2, dtype=int):
                    result[i] = np.moveaxis(result[i], axis, dest)
                    dest += 1
                # reshape
                newshape = [2**nx]
                for count in range(nx-1): 
                    newshape.append(2)
                newshape.append(2**nx)
                result[i] = np.reshape(result[i], newshape)

            elif row % 2 == 1:
                result.append(spin)
                for j in np.arange(1, nx, 1, dtype=int):
                    result[i] = np.tensordot(result[i], spin, axes=0)
                # rearrage axes and reshape
                # original axes order:  vir0 phy vir1 ... vir0 phy vir1
                # new axes order:       (vir0 ... vir0) phy ... phy (vir1 ... vir1)
                # move vir0 (upward) legs to front
                dest = 0
                for axis in np.arange(0, len(result[i].shape), 3, dtype=int):
                    result[i] = np.moveaxis(result[i], axis, dest)
                    dest += 1
                # current axes order: vir0 ... vir0 / phy vir1 ... phy vir1
                # move phy legs to center
                dest = nx
                for axis in np.arange(nx, len(result[i].shape), 2, dtype=int):
                    result[i] = np.moveaxis(result[i], axis, dest)
                    dest += 1
                # reshape
                newshape = [2**nx]
                for count in range(nx): 
                    newshape.append(2)
                newshape.append(2**nx)
                result[i] = np.reshape(result[i], newshape)
            i += 1

    elif xperiodic == True:
        for row in range(2 * ny - 1):
            if (row == 0):
                result.append(edge)
                for j in range(nx - 2):
                    result[i] = np.tensordot(result[i], spin, ([-1],[0]))
                    result[i] = np.tensordot(result[i], edge, ([-1],[0]))
                # take trace due to x-PBC
                result[i] = np.tensordot(result[i], spin, ([0,-1],[-1,0]))
                # original axes order:  vir phy vir phy ... vir phy
                # new axes order:       phy ... phy (vir ... vir)
                # move phy legs to front
                dest = 0
                for axis in np.arange(1, len(result[i].shape), 2, dtype=int):
                    result[i] = np.moveaxis(result[i], axis, dest)
                    dest += 1
                # reshape
                newshape = [1]
                for count in range(nx-1):
                    newshape.append(2)
                newshape.append(2**(nx-1))
                result[i] = np.reshape(result[i], newshape)

            elif (row == 2 * ny - 2):
                result.append(edge)
                for j in range(nx - 2):
                    result[i] = np.tensordot(result[i], spin, ([-1],[0]))
                    result[i] = np.tensordot(result[i], edge, ([-1],[0]))
                # take trace due to x-PBC
                result[i] = np.tensordot(result[i], spin, ([0,-1],[-1,0]))
                # original axes order:  vir phy vir phy ... vir phy
                # new axes order:       vir ... vir (phy ... phy)
                # move vir legs to front
                dest = 0
                for axis in np.arange(0, len(result[i].shape), 2, dtype=int):
                    result[i] = np.moveaxis(result[i], axis, dest)
                    dest += 1
                # reshape
                newshape = [2**(nx-1)]
                for count in range(nx-1):
                    newshape.append(2)
                newshape.append(1)
                result[i] = np.reshape(result[i], newshape)

            elif row % 2 == 0:
                result.append(vertex)
                for j in range(nx - 2):
                    result[i] = np.tensordot(result[i], spin, ([-1],[0]))
                    result[i] = np.tensordot(result[i], vertex, ([-1],[0]))
                # take trace due to x-PBC
                result[i] = np.tensordot(result[i], spin, ([0,-1],[-1,0]))
                # original axes order:  vir0 vir1 phy ... vir0 vir1 phy [(nx - 1) triples]
                # new axes order:       (vir0 ... vir0) phy ... phy (vir1 ... vir1)
                # move vir0 (upward) legs to front
                dest = 0
                for axis in np.arange(0, len(result[i].shape), 3, dtype=int):
                    result[i] = np.moveaxis(result[i], axis, dest)
                    dest += 1
                # current axes order: vir0 ... vir0 / vir1 phy ... vir1 phy
                # position:            0       nx-2   nx-1 nx
                # move phy legs to center
                dest = nx - 1
                for axis in np.arange(nx, len(result[i].shape), 2, dtype=int):
                    result[i] = np.moveaxis(result[i], axis, dest)
                    dest += 1
                # reshape
                newshape = [2**(nx-1)]
                for count in range(nx-1):
                    newshape.append(2)
                newshape.append(2**(nx-1))
                result[i] = np.reshape(result[i], newshape)

            elif row % 2 == 1:
                result.append(spin)
                for j in range(nx - 2):
                    result[i] = np.tensordot(result[i], spin, axes=0)
                # rearrage axes and reshape
                # original axes order:  vir0 phy vir1 ... vir0 phy vir1 [(nx - 1) triples]
                # new axes order:       (vir0 ... vir0) phy ... phy (vir1 ... vir1)
                # move vir0 (upward) legs to front
                dest = 0
                for axis in np.arange(0, len(result[i].shape), 3, dtype=int):
                    result[i] = np.moveaxis(result[i], axis, dest)
                    dest += 1
                # current axes order: vir0 ... vir0 / phy vir1 ... phy vir1
                # position:            0       nx-2   nx-1 nx
                # move phy legs to center
                dest = nx - 1
                for axis in np.arange(nx - 1, len(result[i].shape), 2, dtype=int):
                    result[i] = np.moveaxis(result[i], axis, dest)
                    dest += 1
                # reshape
                newshape = [2**(nx-1)]
                for count in range(nx-1):
                    newshape.append(2)
                newshape.append(2**(nx-1))
                result[i] = np.reshape(result[i], newshape)
            i += 1
    
    # compress MPS
    psi = []
    print("Making ground state")
    for i in tqdm(range(len(result))):
        m = len(result[i].shape[1:-1])
        if m != 1:
            decomp = mps.svd_nsite(m, result[i], 'Fromleft', args)
            for site in range(m):
                psi.append(decomp[site])
        else:
            decomp = copy.copy(result[i])
            psi.append(decomp)

    psi = mps.position(psi, len(psi)-1, args=args)
    psi = mps.normalize(psi, args=args)

    return psi
