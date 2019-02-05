# 
#   gateTEvol.py
#   Toric_Code-Python
#   generate time-evolution gates and swap gates
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

def svd_2site(tensor, cutoff):
    """ Do SVD to decomposite one large tensor into 2 site tensors"""

    virDim = [tensor.shape[0], tensor.shape[-1]]
    phyDim = [tensor.shape[1], tensor.shape[3]]
    mat = np.reshape(tensor, (virDim[0] * phyDim[0]**2, virDim[1] * phyDim[1]**2))

    u,s,v = np.linalg.svd(mat)
    s = s[np.where(np.abs(s[:]) > cutoff)[0]]
    retain_dim = s.shape[0]
    mat_s = np.diag(s)
    u = np.dot(u[:, 0:retain_dim], np.sqrt(mat_s))
    u = np.reshape(u, (virDim[0], phyDim[0], phyDim[0], retain_dim))
    v = np.dot(np.sqrt(mat_s), v[0:retain_dim, :])
    v = np.reshape(v, (retain_dim, phyDim[1], phyDim[1], virDim[1]))
    return u, v

def position(mpo, pos, cutoff):
    """set the orthogonality center of the MPO"""

    siteNum = len(mpo)
    # left normalization
    for i in range(pos):
        virDim = [mpo[i].shape[0], mpo[i].shape[3]]
        phyDim = [mpo[i].shape[1], mpo[i].shape[2]]
        # i,j: virtual leg; a,b: physical leg
        mat = np.einsum('iabj->abij', mpo[i])
        mat = np.reshape(mpo[i], (phyDim[0]*phyDim[1]*virDim[0], virDim[1]))
        a,s,v = np.linalg.svd(mat)
        s = s[np.where(np.abs(s[:]) > cutoff)[0]]
        retain_dim = s.shape[0]
        a = a[:, 0:retain_dim]
        # replace mpo[i] with a
        a = np.reshape(a, (phyDim[0],phyDim[1],virDim[0],retain_dim))
        a = np.einsum('abis->iabs', a)
        mpo[i] = a
        # update mpo[i+1]
        mat_s = np.diag(s)
        v = np.dot(mat_s, v[0:retain_dim, :])
        mpo[i+1] = np.einsum('si,iabj->sabj', v, mpo[i+1])

    # right normalization
    for i in np.arange(siteNum-1, pos, -1, dtype=int):
        virDim = [mpo[i].shape[0], mpo[i].shape[3]]
        phyDim = [mpo[i].shape[1], mpo[i].shape[2]]
        # i,j: virtual leg; a,b: physical leg
        mat = np.einsum('iabj->ijab', mpo[i])
        mat = np.reshape(mpo[i], (virDim[0], virDim[1]*phyDim[0]*phyDim[1]))
        u,s,b = np.linalg.svd(mat)
        s = s[np.where(np.abs(s[:]) > cutoff)[0]]
        retain_dim = s.shape[0]
        b = b[0:retain_dim, :]
        # replace mpo[i] with b
        b = np.reshape(b, (retain_dim,virDim[1],phyDim[0],phyDim[1]))
        b = np.einsum('sjab->sabj', b)
        mpo[i] = b
        # update mpo[i-1]
        mat_s = np.diag(s)
        u = np.dot(u[:, 0:retain_dim], mat_s)
        mpo[i-1] = np.einsum('iabj,js->iabs', mpo[i-1], u)

def gateTEvol(mpo, gateList, ttotal, tstep):
    """
    Perform time evolution to MPO using Trotter gates

    Parameters
    ----------
    mpo : list of numpy arrays
        the MPO to be acted on
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
            gate = gateList[g]
            sites = gate.sites
            # swap gate
            if len(sites) == 2:
                gate2 = copy.copy(gate)
                gate2.gate = np.conj(gate.gate)
                position(mpo, sites[0], 1.0E-10)
                # contraction
                ten_AA = np.einsum('ibek,kdgj,abcd,efgh->iafchj',mpo[sites[0]],mpo[sites[1]],gate.gate,gate2.gate)
                # do svd to restore 2 sites
                m1, m2 = svd_2site(ten_AA, 1.0E-10)
                mpo[sites[0]] = m1
                mpo[sites[1]] = m2
            # time evolution gate
            elif len(sites) == 4:
                gate2 = copy.copy(gate)
                gate2.gate = np.conj(gate.gate)
                position(mpo, sites[1], 1.0E-10)
                # contraction
                ten_AAAA = np.einsum('ibpk,kdrl,lfum,mhwj->ibpdrfuhwj',mpo[sites[0]],mpo[sites[1]],mpo[sites[2]],mpo[sites[3]])
                ten_AAAA = np.einsum('ibpdrfuhwj,abcdefgh,pqrsuvwx->iaqcsevgxj',ten_AAAA,gate.gate,gate2.gate)
                # combine 4 sites into 2 sites
                ten_AAAA = np.reshape(ten_AAAA, (ten_AAAA.shape[0],4,4,4,4,ten_AAAA.shape[-1]))
                mm1, mm2 = svd_2site(ten_AAAA, 1.0E-10)
                # replace sites: 
                del mpo[sites[3]]
                del mpo[sites[2]]
                mpo[sites[0]] = mm1
                mpo[sites[1]] = mm2
                # do svd again to restore 4 sites
                position(mpo, sites[0], 1.0E-10)
                mpo[sites[0]] = np.reshape(mpo[sites[0]], (mpo[sites[0]].shape[0],2,2,2,2,mpo[sites[0]].shape[-1]))
                m1, m2 = svd_2site(mpo[sites[0]], 1.0E-10)
                mpo[sites[0]] = m1
                mpo.insert(sites[1], m2)

                position(mpo, sites[2], 1.0E-10)
                mpo[sites[2]] = np.reshape(mpo[sites[2]], (mpo[sites[2]].shape[0], 2,2,2,2,mpo[sites[2]].shape[-1]))
                m3, m4 = svd_2site(mpo[sites[2]], 1.0E-10)
                mpo[sites[2]] = m3
                mpo.insert(sites[3], m4)
            # error handling
            else:
                print('Wrong number of sites')
                sys.exit()
