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

def savePhyDim(op):
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

def position(op, pos, cutoff, bondm):
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
    allPhyDim = savePhyDim(op2)
    op2 = combinePhyLeg(op2)
    op2 = mps.position(op2, pos, cutoff, bondm, scale=True)
    op2 = decombPhyLeg(op2, allPhyDim)
    return op2

def svd_nsite(n, tensor, cutoff, bondm):
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
    """
    if (len(tensor.shape) != 2*n + 2):
        sys.exit('Wrong dimension of input tensor')
    virDim = [tensor.shape[0], tensor.shape[-1]]
    phyDim = list(tensor.shape[1:-1])
    old_retain_dim = virDim[0]
    result = []
    mat = copy.copy(tensor)
    for i in np.arange(0, n - 1, 1, dtype=int):
        mat = np.reshape(mat, (old_retain_dim * phyDim[2*i] * phyDim[2*i+1], 
        virDim[1] * np.prod(phyDim[2*i+2 : len(phyDim)])))
        u,s,v = np.linalg.svd(mat, full_matrices=False)
        u,s,v,new_retain_dim = mps.svd_truncate(u, s, v, cutoff, bondm, scale=False)
        mat_s = np.diag(s)
        u = np.dot(u, np.sqrt(mat_s))
        v = np.dot(np.sqrt(mat_s), v)
        u = np.reshape(u, (old_retain_dim, phyDim[2*i], phyDim[2*i+1], new_retain_dim))
        result.append(u)
        mat = copy.copy(v)
        old_retain_dim = new_retain_dim
    v = np.reshape(v, (old_retain_dim, phyDim[-2], phyDim[-1], virDim[-1]))
    result.append(v)
    return result

def gateTEvol(op, gateList, ttotal, tstep, cutoff, bondm):
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
    cutoff: float
        largest value of s_max/s_min
    bondm : int
        largest virtual bond dimension allowed
    """
    op2 = copy.copy(op)
    # number of steps
    nt = int(ttotal/tstep + (1e-9 * (ttotal/tstep)))
    if (np.abs(nt*tstep-ttotal) > 1.0E-9): 
        print("Timestep not commensurate with total time")
        sys.exit()
    gateNum = len(gateList)

    for tt in range(nt):
        for g in range(gateNum):
            gate2 = gateList[g].gate
            sites = gateList[g].sites
            gate = np.conj(gate2)
            # field gate
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
                result = svd_nsite(2, ten_AA, cutoff, bondm)
                for i in range(2):
                    op2[sites[i]] = result[i]
            # time evolution gate
            elif len(sites) == 4:
                # gauging and normalizing
                # op2 = position(op2, sites[1], 10, cutoff, bondm)
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
                result = svd_nsite(4, ten_AAAA, cutoff, bondm)
                for i in range(4):
                    op2[sites[i]] = result[i]
            # error handling
            else:
                sys.exit('Wrong number of sites of gate')
    
    return op2

def sum(mpo1, mpo2, cutoff, bondm):
    """
    Calculate the inner product of two MPO's

    Parameters
    ------------
    mpo1, mpo2 : list of numpy arrays
        The two MPOs to be added
    cutoff : float
        largest value of s_max/s_min
    bondm : int
        largest virtual bond dimension allowed
    """
    # dimension check
    if len(mpo1) != len(mpo2):
        sys.exit("The lengths of the two MPOs do not match")
    for i in range(len(mpo1)):
        if (mpo1[i].shape[1] != mpo2[i].shape[1] or mpo1[i].shape[2] != mpo2[i].shape[2]):
            sys.exit("The physical dimensions of the two MPOs do not match")
    result = []
    for i in range(len(mpo1)):
        virDim1 = [mpo1[i].shape[0], mpo1[i].shape[-1]]
        virDim2 = [mpo2[i].shape[0], mpo2[i].shape[-1]]
        phyDim = [mpo1[i].shape[1], mpo1[i].shape[2]]
        virDim = [virDim1[0]+virDim2[0], virDim1[1]+virDim2[1]]
        result.append(np.zeros((virDim[0], phyDim[0], phyDim[1], virDim[1]), dtype=complex))
        # direct sum
        result[i][0:virDim1[0],:,:,0:virDim1[1]] = mpo1[i]
        result[i][virDim1[0]:virDim[0],:,:,virDim1[1]:virDim[1]] = mpo2[i]
    result = position(result, 0, cutoff, bondm)
    return result

def save_to_file(op, filename):
    """
    Save MPO (shape and nonzero elements) to (txt) file

    Parameters
    ----------
    op : list of numpy arrays
        the MPO to be written to file
    filename : string
        name of the output file
    """
    with open(filename, 'w+') as f:
        for i in range(len(op)):
            f.write(str(i) + '\t')
            f.write(str(op[i].shape[0]) + '\t')
            f.write(str(op[i].shape[1]) + '\t')
            f.write(str(op[i].shape[2]) + '\t')
            f.write(str(op[i].shape[3]) + '\n')
            for m,n,p,q in product(range(op[i].shape[0]),range(op[i].shape[1]),range(op[i].shape[2]),range(op[i].shape[3])):
                if op[i][m,n,p,q] != 0:
                    f.write(str(m) + '\t')
                    f.write(str(n) + '\t')
                    f.write(str(p) + '\t')
                    f.write(str(q) + '\t')
                    f.write(str(op[i][m,n,p,q]) + '\n')
            # separation line consisting of -1
            f.write(str(-1) + '\t' + str(-1) + '\t' + str(-1) + '\t' + str(-1) + '\t' + str(-1) + '\n')
    