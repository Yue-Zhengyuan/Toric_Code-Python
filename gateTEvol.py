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

global_cutoff = 1.0E-10

def svd_2site(tensor, cutoff):
    """
    Do SVD to decomposite one large tensor into 2 site tensors of an MPS
    
    Parameters
    ---------------
    tensor : numpy array
        the 2-site tensor to be decomposed
    cutoff: float
        precision for SVD cutoff
    """

    virDim = [tensor.shape[0], tensor.shape[-1]]
    phyDim = tensor.shape[1]
    mat = np.reshape(tensor, (virDim[0] * phyDim, virDim[1] * phyDim))

    u,s,v = np.linalg.svd(mat)
    s = s[np.where(np.abs(s[:]) > cutoff)[0]]
    retain_dim = s.shape[0]
    mat_s = np.diag(s)
    u = np.dot(u[:, 0:retain_dim], np.sqrt(mat_s))
    u = np.reshape(u, (virDim[0], phyDim[0], phyDim[0], retain_dim))
    v = np.dot(np.sqrt(mat_s), v[0:retain_dim, :])
    v = np.reshape(v, (retain_dim, phyDim[1], phyDim[1], virDim[1]))
    return u, v

def position(mps, pos, cutoff):
    """
    set the orthogonality center of the MPS
    
    Parameters
    ----------------
    mps : list of numpy arrays
        the MPS to be acted on
    pos : int
        the position of the orthogonality center
    cutoff: float
        precision for SVD cutoff
    """

    siteNum = len(mps)
    # left normalization
    for i in range(pos):
        virDim = [mps[i].shape[0], mps[i].shape[-1]]
        phyDim = mps[i].shape[1]
        # i,j: virtual leg; a: physical leg
        mat = np.einsum('iaj->aij', mps[i])
        mat = np.reshape(mps[i], (phyDim*virDim[0], virDim[1]))
        a,s,v = np.linalg.svd(mat)
        s = s[np.where(np.abs(s[:]) > cutoff)[0]]
        retain_dim = s.shape[0]
        a = a[:, 0:retain_dim]
        # replace mps[i] with a
        a = np.reshape(a, (phyDim,virDim[0],retain_dim))
        a = np.einsum('ais->ias', a)
        mps[i] = a
        # update mps[i+1]
        mat_s = np.diag(s)
        v = np.dot(mat_s, v[0:retain_dim, :])
        mps[i+1] = np.einsum('si,iaj->saj', v, mps[i+1])

    # right normalization
    for i in np.arange(siteNum-1, pos, -1, dtype=int):
        virDim = [mps[i].shape[0], mps[i].shape[-1]]
        phyDim = mps[i].shape[1]
        # i,j: virtual leg; a: physical leg
        mat = np.einsum('iaj->ija', mps[i])
        mat = np.reshape(mps[i], (virDim[0], virDim[1]*phyDim))
        u,s,b = np.linalg.svd(mat)
        s = s[np.where(np.abs(s[:]) > cutoff)[0]]
        retain_dim = s.shape[0]
        b = b[0:retain_dim, :]
        # replace mps[i] with b
        b = np.reshape(b, (retain_dim,virDim[1],phyDim))
        b = np.einsum('sja->saj', b)
        mps[i] = b
        # update mps[i-1]
        mat_s = np.diag(s)
        u = np.dot(u[:, 0:retain_dim], mat_s)
        mps[i-1] = np.einsum('iaj,js->ias', mps[i-1], u)

def gateTEvol(mps, gateList, ttotal, tstep):
    """
    Perform time evolution to MPS using Trotter gates

    Parameters
    ----------
    mps : list of numpy arrays
        the MPS to be acted on
    gateList : list of gates
        gates used in one evolution step
    ttotal : real float number
        total time of evolution
    tstep : real float number
        time of each evolution step
    """

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
                position(mps, sites[0], global_cutoff)
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
                ten_AA = np.einsum('ibk,kdj,abcd->iacj',mps[sites[0]],mps[sites[1]],gate)
                # do svd to restore 2 sites
                m1, m2 = svd_2site(ten_AA, global_cutoff)
                mps[sites[0]] = m1
                mps[sites[1]] = m2
            # time evolution gate
            elif len(sites) == 4:
                position(mps, sites[1], global_cutoff)
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
                ten_AAAA = np.einsum('ibj,jdk,kfl,lhm->ibdfhm',mps[sites[0]],mps[sites[1]],mps[sites[2]],mps[sites[3]])
                ten_AAAA = np.einsum('ibdfhm,abcdefgh->iacegm',ten_AAAA,gate)
                # combine 4 sites into 2 sites
                ten_AAAA = np.reshape(ten_AAAA, (ten_AAAA.shape[0],4,4,ten_AAAA.shape[-1]))
                mm1, mm2 = svd_2site(ten_AAAA, global_cutoff)
                # replace sites: 
                del mps[sites[3]]
                del mps[sites[2]]
                mps[sites[0]] = mm1
                mps[sites[1]] = mm2
                # do svd again to restore 4 sites
                position(mps, sites[0], global_cutoff)
                mps[sites[0]] = np.reshape(mps[sites[0]], (mps[sites[0]].shape[0],2,2,mps[sites[0]].shape[-1]))
                m1, m2 = svd_2site(mps[sites[0]], global_cutoff)
                mps[sites[0]] = m1
                mps.insert(sites[1], m2)

                position(mps, sites[2], global_cutoff)
                mps[sites[2]] = np.reshape(mps[sites[2]], (mps[sites[2]].shape[0],2,2,mps[sites[2]].shape[-1]))
                m3, m4 = svd_2site(mps[sites[2]], global_cutoff)
                mps[sites[2]] = m3
                mps.insert(sites[3], m4)
            # error handling
            else:
                print('Wrong number of sites')
                sys.exit()
