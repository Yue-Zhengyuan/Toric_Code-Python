# 
#   mpo.py
#   Toric_Code-Python
#   Operations related to MPO
#
#   created on Feb 11, 2019 by Yue Zhengyuan
#

import numpy as np
# import scipy.linalg as LA
import sys
from copy import copy
import mps
import para_dict as p
from itertools import product

def combinePhyLeg(op):
    """
    Combine the physical legs of an MPO to convert it into MPS
    """
    psi = copy(op)
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
    op = copy(psi)
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

def position(op, pos, args, oldcenter=-1, compute_entg=False):
    """
    set the orthogonality center of the MPO
    with respect to only part of the MPO
    
    Parameters
    ----------------
    oldcenter : int between -1 and len(psi) - 1 (default = -1)
        when old center <= 0,            do right canonization
        when old center == len(psi)-1,   do left canonization
    compute_entg : default False
        if True: return the entanglement entropy between the two sides of the orthogonality center
    """
    op2 = copy(op)
    allPhyDim = getPhyDim(op2)
    op2 = combinePhyLeg(op2)
    if compute_entg == True:
        op2, entg = mps.position(op2, pos, args, oldcenter=oldcenter, 
        preserve_norm=True, compute_entg=True)
        op2 = decombPhyLeg(op2, allPhyDim)
        return op2, entg
    else:
        op2 = mps.position(op2, pos, args, oldcenter=oldcenter, 
        preserve_norm=True, compute_entg=False)
        op2 = decombPhyLeg(op2, allPhyDim)
        return op2

def svd_nsite(n, tensor, dir, args):
    """
    Do SVD to decomposite one large tensor into n site tensors of an MPO
    
    Parameters
    ---------------
    dir: 'Fromleft'/'Fromright'
        if dir == 'Fromleft'/'Fromright', the last/first of the n sites will be orthogonality center of the n tensors
    args : dict
        parameters controlling SVD
    preserve_norm : default True 
        determine how s is combined with u,v in SVD
        if True, uniformly distribute the norm of tensor among the resulting matrices
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

    result = mps.svd_nsite(n, newtensor, dir, args, preserve_norm=True)
    # restore physical legs
    for i in range(n):
        newshape = (result[i].shape[0], phyDim[2*i], 
        phyDim[2*i+1], result[i].shape[-1])
        result[i] = np.reshape(result[i], newshape)
    return result

def gateTEvol(op, gateList, ttotal, tstep, args):
    """
    Perform time evolution to MPO using Trotter gates
    Heisenberg picture: O(dt) = exp(+iH dt) O(0) exp(-iH dt)

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
    op2 = copy(op)
    nt = int(ttotal/tstep + (1e-9 * (ttotal/tstep)))
    if (np.abs(nt*tstep-ttotal) > 1.0E-9): 
        print("Timestep not commensurate with total time")
        sys.exit()
    gateNum = len(gateList)

    op2 = position(op2, gateList[0].sites[0], args=args)
    oldcenter = 0
    for tt, g in product(range(nt), range(gateNum)):
        gate2 = gateList[g].gate
        sites = gateList[g].sites
        # if gateList[g].sites[-1] < gateList[g].sites[0]:
        #     print("debug\n")
        gate = np.conj(gate2)
        if len(sites) == 2:
            # contraction
            #
            #       a      c
            #      _|______|_
            #      |        |       gate = exp(+iH dt)
            #      -|------|-
            #       b      d
            #      _|_    _|_
            #  i --| |-k--| |--j    op
            #      -|-    -|-
            #       e      g
            #      _|______|_
            #      |        |       gate2 = exp(-iH dt)
            #      -|------|-
            #       f      h
            #
            ten_AA = np.einsum('ibek,kdgj,abcd->iaecgj',
            op2[sites[0]], op2[sites[1]], gate, optimize=True)
            ten_AA = np.einsum('iaecgj,efgh->iafchj',ten_AA,gate2, optimize=True)
            # do svd to restore 2 sites
            if g < gateNum - 1:
                if gateList[g+1].sites[0] >= gateList[g].sites[-1]:
                    result = svd_nsite(2, ten_AA, 'Fromleft', args=args)
                    for i in range(2):
                        op2[sites[i]] = result[i]
                    oldcenter = sites[-1]
                    op2 = position(op2, gateList[g+1].sites[0], args, oldcenter=oldcenter)
                    oldcenter = gateList[g+1].sites[0]
                else:
                    result = svd_nsite(2, ten_AA, args=args, dir='Fromright')
                    for i in range(2):
                        op2[sites[i]] = result[i]
                    oldcenter = sites[0]
                    op2 = position(op2, gateList[g+1].sites[-1], args, oldcenter=oldcenter)
                    oldcenter = gateList[g+1].sites[-1]
            else:
                result = svd_nsite(2, ten_AA, 'Fromright', args=args)
                for i in range(2):
                    op2[sites[i]] = result[i]
                oldcenter = sites[0]
                op2 = position(op2, 0, args, oldcenter=oldcenter)
                oldcenter = 0
        elif len(sites) == 4:
            # contraction
            #
            #       a      c      e      g
            #      _|______|______|______|_
            #      |                      |         gate = exp(+iH dt)
            #      -|------|------|------|-
            #       b      d      f      h
            #      _|_    _|_    _|_    _|_
            #  i --| |-j--| |--k-| |-l--| |-- m
            #      -|-    -|-    -|-    -|-
            #       s      u      w      y
            #      _|______|______|______|_
            #      |                      |         gate2 = exp(-iH dt)
            #      -|------|------|------|-
            #       t      v      x      z
            #
            ten_AAAA = np.einsum('ibsj,jduk,kfwl,lhym->ibsdufwhym',
            op2[sites[0]],op2[sites[1]],op2[sites[2]],op2[sites[3]], optimize=True)
            ten_AAAA = np.einsum('abcdefgh,ibsdufwhym->iascuewgym',
            gate, ten_AAAA, optimize=True)
            ten_AAAA = np.einsum('iascuewgym,stuvwxyz->iatcvexgzm',
            ten_AAAA, gate2, optimize=True)
            if g < gateNum - 1:
                if gateList[g+1].sites[0] >= gateList[g].sites[-1]:
                    result = svd_nsite(4, ten_AAAA, 'Fromleft', args=args)
                    for i in range(4):
                        op2[sites[i]] = result[i]
                    oldcenter = sites[-1]
                    op2 = position(op2, gateList[g+1].sites[0], args, oldcenter=oldcenter)
                    oldcenter = gateList[g+1].sites[0]
                else:
                    result = svd_nsite(4, ten_AAAA, 'Fromright',args=args)
                    for i in range(4):
                        op2[sites[i]] = result[i]
                    oldcenter = sites[0]
                    op2 = position(op2, gateList[g+1].sites[-1], args, oldcenter=oldcenter)
                    oldcenter = gateList[g+1].sites[-1]
            else:
                result = svd_nsite(4, ten_AAAA, 'Fromright', args=args)
                for i in range(4):
                    op2[sites[i]] = result[i]
                oldcenter = sites[0]
                op2 = position(op2, 0, args, oldcenter=oldcenter)
                oldcenter = 0
        # error handling
        else:
            sys.exit('Wrong number of sites of gate')
    return op2

def sum(op1, op2, args, compress="svd"):
    """
    Calculate the sum of two MPO's

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

    if (compress == None or compress == "svd"):
        psi1 = copy(op1)
        allPhyDim = getPhyDim(op1)
        psi1 = combinePhyLeg(psi1)

        psi2 = copy(op2)
        psi2 = combinePhyLeg(psi2)
        result = mps.sum(psi1, psi2, args, compress=compress)
        result = decombPhyLeg(result, allPhyDim)

    # variational optimization
    elif (compress == "var"):
        pass
    
    else: 
        sys.exit("Wrong compress parameter")
    return result

def prod(op1, op2, args, compress="svd"):
    """
    Calculate the product of two MPO's

    Parameters
    ------------
    op1, op2 : list of numpy arrays
        The two MPOs to be multiplied
    args : dict
        parameters controlling SVD
    """
    # dimension check
    if len(op1) != len(op2):
        sys.exit("The lengths of the two MPOs do not match")
    for i in range(len(op1)):
        if (op1[i].shape[1] != op2[i].shape[2]):
            sys.exit("The physical dimensions of the two MPOs do not match")

    result = []
    for i in range(len(op1)):
        result.append(np.tensordot(op1[i], op2[i], axes=([1],[2])))
        result[i] = np.transpose(result[i], axes=(1,3,5,0,2,4))
        shape = np.shape(result[i])
        result[i] = np.reshape(result[i], 
        (shape(0)*shape(1), shape(2)*shape(3), shape(4)*shape(5)))
        
    if compress == None:
        pass
    # svd compress
    elif compress == "svd":
        result = position(result, 0, args)
    # variational optimization
    elif (compress == "var"):
        pass
    else: 
        sys.exit("Wrong compress parameter")
    return result

def printdata(op):
    mps.printdata(op)

def save_to_file(op, filename):
    """
    Save MPO (shape and nonzero elements) to (txt) file
    """
    mps.save_to_file(op, filename)
    