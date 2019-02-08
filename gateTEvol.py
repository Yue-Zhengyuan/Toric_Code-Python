# 
#   gateTEvol.py
#   Toric_Code-Python
#   Time-evolution of MPS with Trotter and swap gates
#
#   created on Jan 21, 2019 by Yue Zhengyuan
#   code influcenced by iTensor - tevol.h
#

import numpy as np
import sys
from itertools import product
import para_dict as p
import copy
import gates
import mps

def svd_2site(tensor, cutoff, bondm):
    """
    Do SVD to decomposite one large tensor into 2 site tensors of an MPS
    
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
    phyDim = tensor.shape[1]
    mat = np.reshape(tensor, (virDim[0] * phyDim, virDim[1] * phyDim))

    u,s,v = np.linalg.svd(mat)
    u,s,v,retain_dim = mps.svd_truncate(u, s, v, cutoff, bondm)
    mat_s = np.diag(s)
    u = np.dot(u, np.sqrt(mat_s))
    v = np.dot(np.sqrt(mat_s), v)
    u = np.reshape(u, (virDim[0], phyDim, retain_dim))
    v = np.reshape(v, (retain_dim, phyDim, virDim[1]))
    return u, v

def gateTEvol(psi, gateList, ttotal, tstep, cutoff, bondm):
    """
    Perform time evolution to MPS using Trotter gates

    Parameters
    ----------
    psi : list of numpy arrays
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
    phi = copy.copy(psi)
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
                mps.position(phi, sites[0], cutoff, bondm)
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
                mps.position(phi, sites[1], cutoff, bondm)
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
                mps.position(phi, sites[0], cutoff, bondm)
                phi[sites[0]] = np.reshape(phi[sites[0]], (phi[sites[0]].shape[0],2,2,phi[sites[0]].shape[-1]))
                m1, m2 = svd_2site(phi[sites[0]], cutoff, bondm)
                phi[sites[0]] = m1
                phi.insert(sites[1], m2)

                mps.position(phi, sites[2], cutoff, bondm)
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
