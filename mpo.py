# 
#   mpo.py
#   Toric_Code-Python
#   Operations related to MPO
#
#   created on Feb 11, 2019 by Yue Zhengyuan
#

import numpy as np
import sys
import copy
import mps
import gates
from itertools import product

def combinePhyLeg(op):
    """
    Combine the physical legs of an MPO to convert it into MPS
    """
    psi = copy.copy(op)
    for i in range(len(op)):
        virDim = (op[i].shape[0], op[i].shape[-1])
        phyDim = op[i].shape[1:-1]
        psi[i] = np.reshape(psi[i], (virDim[0], np.prod(phyDim), virDim[1]))
    return psi

def getPhyDim(op):
    """
    Save the physical leg dimension of an MPO
    """
    allPhyDim = []
    for i in range(len(op)):
        allPhyDim.append(op[i].shape[1:-1])
    return allPhyDim

def decombPhyLeg(psi, allPhyDim):
    """
    Decombine the physical legs of an MPS to convert it into MPO

    Parameters
    -------------
    allPhyDim: list of two-arrays
        Physical leg dimension of the MPO on each site
    """
    op = copy.copy(psi)
    for i in range(len(op)):
        virDim = (op[i].shape[0], op[i].shape[-1])
        phyDim = allPhyDim[i]
        newshape = []
        newshape.append(virDim[0])
        for dim in phyDim:
            newshape.append(dim)
        newshape.append(virDim[1])
        op[i] = np.reshape(op[i], tuple(newshape))
    return op

def position(op, pos, args={'cutoff':1.0E-5, 'bondm':200, 'scale':True}):
    """
    set the orthogonality center of the MPO
    with respect to only part of the MPO
    
    Parameters
    ----------------
    op : list of numpy arrays
        the MPO to be acted on
    pos : int
        the position of the orthogonality center
    cutoff : float
        largest value of s_max/s_min
    bondm : int
        largest virtual bond dimension allowed
    """
    op2 = copy.copy(op)
    allPhyDim = getPhyDim(op2)
    op2 = combinePhyLeg(op2)
    op2 = mps.position(op2, pos, args=args)
    op2 = decombPhyLeg(op2, allPhyDim)
    return op2

def svd_nsite(n, tensor, dir, args={'cutoff':1.0E-5, 'bondm':200, 'scale':True}):
    """
    Do SVD to decomposite one large tensor into n site tensors of an MPO
    
    Parameters
    ---------------
    n : int
        number of site tensors to be produces
    tensor : numpy array
        the 2-site tensor to be decomposed
    cutoff : float
        largest value of s_max/s_min
    bondm : int
        largest virtual bond dimension allowed
    dir: 'Fromleft'(default)/'Fromright'
        if dir == 'left'/'right', the last/first of the n sites will be orthogonality center
    """
    if (len(tensor.shape) != 2*n + 2):
        sys.exit('Wrong dimension of input tensor')
    # combine physical legs to convert MPO to MPS
    virDim = [tensor.shape[0], tensor.shape[-1]]
    phyDim = list(tensor.shape[1:-1])
    newshape = [virDim[0]]
    for i in np.arange(0, len(phyDim), 2, dtype=int):
        newshape.append(phyDim[i] * phyDim[i+1])
    newshape.append(virDim[1])
    newtensor = np.reshape(tensor, newshape)

    result = mps.svd_nsite(n, newtensor, dir, args=args)
    # restore physical legs
    for i in range(n):
        newshape = (result[i].shape[0], phyDim[2*i], 
        phyDim[2*i+1], result[i].shape[-1])
        result[i] = np.reshape(result[i], newshape)
    return result

def gateTEvol(op, gateList, ttotal, tstep, 
args={'cutoff':1.0E-5, 'bondm':200, 'scale':True}):
    """
    Perform time evolution to MPO using Trotter gates

    Parameters
    ----------
    op : list of numpy arrays
        the MPS to be acted on
    gateList : list of gates
        gates used in one evolution step
    ttotal : real float number
        total time of evolution
    tstep : real float number
        time of each evolution step
    args : dict
        parameters controlling SVD
    """
    op2 = copy.copy(op)
    nt = int(ttotal/tstep + (1e-9 * (ttotal/tstep)))
    if (np.abs(nt*tstep-ttotal) > 1.0E-9): 
        print("Timestep not commensurate with total time")
        sys.exit()
    gateNum = len(gateList)

    op2 = position(op2, gateList[0].sites[0], args=args)
    for tt in range(nt):
        for g in range(gateNum):
            gate2 = gateList[g].gate
            sites = gateList[g].sites
            gate = np.conj(gate2)
            if len(sites) == 1:
                # op2 = position(op2, sites[0], 10, cutoff, bondm)
                # contraction
                #
                #       a
                #      _|_
                #      | |      gate = exp(+i H dt)
                #      -|-
                #       b
                #      _|_
                #  i --| |-- k
                #      -|- 
                #       e
                #      _|_
                #      | |      gate2 = exp(-i H dt)
                #      -|-
                #       f
                #
                ten_AA = np.einsum('ibek,ab->iaek',op2[sites[0]],gate)
                ten_AA = np.einsum('iaek,ef->iafk',ten_AA,gate2)
                op2[sites[0]] = ten_AA
                if g < gateNum - 1:
                    op2 = position(op2, gateList[g+1].sites[0], args=args)
                else:
                    op2 = position(op2, 0, args=args)
            # swap gate
            elif len(sites) == 2:
                # op2 = position(op2, sites[0], 10, cutoff, bondm)
                # contraction
                #
                #       a      c
                #      _|______|_
                #      |        |
                #      -|------|-
                #       b      d
                #      _|_    _|_
                #  i --| |-k--| |--j
                #      -|-    -|-
                #       e      g
                #      _|______|_
                #      |        |
                #      -|------|-
                #       f      h
                #
                ten_AA = np.einsum('ibek,kdgj,abcd->iaecgj',op2[sites[0]],op2[sites[1]],gate)
                ten_AA = np.einsum('iaecgj,efgh->iafchj',ten_AA,gate2)
                # do svd to restore 2 sites
                if g < gateNum - 1:
                    if gateList[g+1].sites[0] >= gateList[g].sites[-1]:
                        result = svd_nsite(2, ten_AA, 'Fromleft', args=args)
                        for i in range(2):
                            op2[sites[i]] = result[i]
                        op2 = position(op2, gateList[g+1].sites[0], args=args)
                    else:
                        result = svd_nsite(2, ten_AA, args=args, dir='Fromright')
                        for i in range(2):
                            op2[sites[i]] = result[i]
                        op2 = position(op2, gateList[g+1].sites[-1], args=args)
                else:
                    result = svd_nsite(2, ten_AA, 'Fromright', args=args)
                    for i in range(2):
                        op2[sites[i]] = result[i]
                    op2 = position(op2, 0, args=args)
            # time evolution gate
            elif len(sites) == 4:
                # contraction
                #
                #       a      c      e      g
                #      _|______|______|______|_
                #      |                      |
                #      -|------|------|------|-
                #       b      d      f      h
                #      _|_    _|_    _|_    _|_
                #  i --| |-j--| |--k-| |-l--| |-- m
                #      -|-    -|-    -|-    -|-
                #       s      u      w      y
                #      _|______|______|______|_
                #      |                      |
                #      -|------|------|------|-
                #       t      v      x      z
                #
                ten_AAAA = np.einsum('ibsj,jduk,kfwl,lhym->ibsdufwhym',op2[sites[0]],op2[sites[1]],op2[sites[2]],op2[sites[3]])
                ten_AAAA = np.einsum('abcdefgh,ibsdufwhym->iascuewgym',gate,ten_AAAA)
                ten_AAAA = np.einsum('iascuewgym,stuvwxyz->iatcvexgzm',ten_AAAA,gate2)
                if g < gateNum:
                    if gateList[g+1].sites[0] >= gateList[g].sites[-1]:
                        result = svd_nsite(4, ten_AAAA, 'Fromleft', args=args)
                        for i in range(4):
                            op2[sites[i]] = result[i]
                        op2 = position(op2, gateList[g+1].sites[0], args=args)
                    else:
                        result = svd_nsite(4, ten_AAAA, 'Fromright',args=args)
                        for i in range(4):
                            op2[sites[i]] = result[i]
                        op2 = position(op2, gateList[g+1].sites[-1], args=args)
                else:
                    result = svd_nsite(4, ten_AAAA, 'Fromright', args=args)
                    for i in range(4):
                        op2[sites[i]] = result[i]
                    op2 = position(op2, 0, args=args)
            # error handling
            else:
                sys.exit('Wrong number of sites of gate')
    
    return op2

def sum(op1, op2, args={'cutoff':1.0E-5, 'bondm':200, 'scale':True}):
    """
    Calculate the inner product of two MPO's

    Parameters
    ------------
    op1, op2 : list of numpy arrays
        The two MPOs to be added
    args : dict
        parameters controlling SVD
    """
    # dimension check
    if len(op1) != len(op2):
        sys.exit("The lengths of the two MPOs do not match")
    for i in range(len(op1)):
        if (op1[i].shape[1] != op2[i].shape[1] or op1[i].shape[2] != op2[i].shape[2]):
            sys.exit("The physical dimensions of the two MPOs do not match")

    psi1 = copy.copy(op1)
    allPhyDim = getPhyDim(op1)
    psi1 = combinePhyLeg(psi1)

    psi2 = copy.copy(op2)
    psi2 = combinePhyLeg(psi2)
    result = mps.sum(psi1, psi2, args=args)
    result = decombPhyLeg(result, allPhyDim)
    return result

def printdata(op):
    mps.printdata(op)

def save_to_file(op, filename):
    """
    Save MPO (shape and nonzero elements) to (txt) file
    """
    mps.save_to_file(op, filename)
    