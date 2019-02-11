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

def position(op, pos, length, cutoff, bondm):
    """
    set the orthogonality center of the MPO
    with respect to only part of the MPO
    
    Parameters
    ----------------
    op : list of numpy arrays
        the MPO to be acted on
    pos : int
        the position of the orthogonality center
    length : int
        length of the part to be gauged
    cutoff : float
        largest value of s_max/s_min
    bondm : int
        largest virtual bond dimension allowed
    """
    op2 = copy.copy(op)
    siteNum = len(op2)
    left = pos - int(length/2)
    right = pos +  int(length/2)
    if left < 0:
        left = 0
        right = length - 1
    elif right > siteNum - 1:
        left = siteNum - 1 - length
        right = siteNum - 1
    # left normalization
    for i in np.arange(left, pos, 1, dtype=int):
        virDim = [op2[i].shape[0], op2[i].shape[-1]]
        phyDim = [op2[i].shape[1], op2[i].shape[2]]
        # i,j: virtual leg; a,b: physical leg
        mat = np.einsum('iabj->abij', op2[i])
        mat = np.reshape(op2[i], (phyDim[0]*phyDim[1]*virDim[0], virDim[1]))
        a,s,v = np.linalg.svd(mat)
        a,s,v,retain_dim = mps.svd_truncate(a, s, v, cutoff, bondm)
        # replace op[i] with a
        a = np.reshape(a, (phyDim[0],phyDim[1],virDim[0],retain_dim))
        a = np.einsum('abis->iabs', a)
        op2[i] = a
        # update op[i+1]
        mat_s = np.diag(s)
        v = np.dot(mat_s, v)
        op2[i+1] = np.einsum('si,iabj->sabj', v, op2[i+1])

    # right normalization
    for i in np.arange(right, pos, -1, dtype=int):
        virDim = [op2[i].shape[0], op2[i].shape[-1]]
        phyDim = [op2[i].shape[1], op2[i].shape[2]]
        # i,j: virtual leg; a,b: physical leg
        mat = np.einsum('iabj->ijab', op2[i])
        mat = np.reshape(op2[i], (virDim[0], virDim[1]*phyDim[0]*phyDim[1]))
        u,s,b = np.linalg.svd(mat)
        u,s,b,retain_dim = mps.svd_truncate(u, s, b, cutoff, bondm)
        # replace mps[i] with b
        b = np.reshape(b, (retain_dim,virDim[1],phyDim[0],phyDim[1]))
        b = np.einsum('sjab->sabj', b)
        op2[i] = b
        # update mps[i-1]
        mat_s = np.diag(s)
        u = np.dot(u, mat_s)
        op2[i-1] = np.einsum('iabj,js->iabs', op2[i-1], u)
    
    return op2

def svd_2site(tensor, cutoff, bondm):
    """
    Do SVD to decomposite one large tensor into 2 site tensors of an MPO
    
    Parameters
    ---------------
    tensor : numpy array
        the 2-site tensor to be decomposed
    cutoff : float
        largest value of s_max/s_min
    bondm : int
        largest virtual bond dimension allowed
    """

    virDim = [tensor.shape[0], tensor.shape[-1]]
    phyDim = [tensor.shape[1], tensor.shape[2]]
    mat = np.reshape(tensor, (virDim[0] * phyDim[0] * phyDim[1], virDim[1] * phyDim[0] * phyDim[1]))

    u,s,v = np.linalg.svd(mat)
    u,s,v,retain_dim = mps.svd_truncate(u, s, v, cutoff, bondm)
    mat_s = np.diag(s)
    u = np.dot(u, np.sqrt(mat_s))
    v = np.dot(np.sqrt(mat_s), v)
    u = np.reshape(u, (virDim[0], phyDim[0], phyDim[1], retain_dim))
    v = np.reshape(v, (retain_dim, phyDim[0], phyDim[1], virDim[1]))
    return u, v

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
    phi = copy.copy(op)
    nt = int(ttotal/tstep + (1e-9 * (ttotal/tstep)))
    if (np.abs(nt*tstep-ttotal) > 1.0E-9): 
        print("Timestep not commensurate with total time")
        sys.exit()
    gateNum = len(gateList)

    for tt in range(nt):
        for g in range(gateNum):
            gate = gateList[g].gate
            sites = gateList[g].sites
            # swap gate
            if len(sites) == 2:
                # gauging and normalizing
                phi = mps.position(phi, sites[0], cutoff, bondm)
                # norm = np.tensordot(phi[sites[0]], np.conj(phi[sites[0]]), ([0,1,2],[0,1,2]))
                # norm = np.sqrt(norm)
                # phi /= norm
                # phi = list(phi)
                # contraction
                #
                #       a      c
                #      _|______|_
                #      |        |
                #      -|------|-
                #       b      d
                #      _|_    _|_
                #  i --| |-k--| |--j
                #      ---    ---
                #
                ten_AA = np.einsum('ibk,kdj,abcd->iacj',phi[sites[0]],phi[sites[1]],gate)
                # do svd to restore 2 sites
                m1, m2 = svd_2site(ten_AA, cutoff, bondm)
                phi[sites[0]] = m1
                phi[sites[1]] = m2
            # time evolution gate
            elif len(sites) == 4:
                # gauging and normalizing
                phi = mps.position(phi, sites[1], cutoff, bondm)
                # norm = np.tensordot(phi[sites[1]], np.conj(phi[sites[1]]), ([0,1,2],[0,1,2]))
                # norm = np.sqrt(norm)
                # phi /= norm
                # phi = list(phi)
                # contraction
                #
                #       a      c      e      g
                #      _|______|______|______|_
                #      |                      |
                #      -|------|------|------|-
                #       b      d      f      h
                #      _|_    _|_    _|_    _|_
                #  i --| |-j--| |--k-| |-l--| |-- m
                #      ---    ---    ---    ---
                #
                ten_AAAA = np.einsum('ibj,jdk,kfl,lhm->ibdfhm',phi[sites[0]],phi[sites[1]],phi[sites[2]],phi[sites[3]])
                ten_AAAA = np.einsum('ibdfhm,abcdefgh->iacegm',ten_AAAA,gate)
                # combine 4 sites into 2 sites
                ten_AAAA = np.reshape(ten_AAAA, (ten_AAAA.shape[0],4,4,ten_AAAA.shape[-1]))
                mm1, mm2 = svd_2site(ten_AAAA, cutoff, bondm)
                # replace sites: 
                del phi[sites[3]]
                del phi[sites[2]]
                phi[sites[0]] = mm1
                phi[sites[1]] = mm2
                # do svd again to restore 4 sites
                phi = mps.position(phi, sites[0], cutoff, bondm)
                phi[sites[0]] = np.reshape(phi[sites[0]], (phi[sites[0]].shape[0],2,2,phi[sites[0]].shape[-1]))
                m1, m2 = svd_2site(phi[sites[0]], cutoff, bondm)
                phi[sites[0]] = m1
                phi.insert(sites[1], m2)

                phi = mps.position(phi, sites[2], cutoff, bondm)
                phi[sites[2]] = np.reshape(phi[sites[2]], (phi[sites[2]].shape[0],2,2,phi[sites[2]].shape[-1]))
                m3, m4 = svd_2site(phi[sites[2]], cutoff, bondm)
                phi[sites[2]] = m3
                phi.insert(sites[3], m4)
            # error handling
            else:
                print('Wrong number of sites')
                sys.exit()
    # return a guaged and normalized MPS |phi>
    phi = mps.normalize(phi, cutoff, bondm)
    return phi
