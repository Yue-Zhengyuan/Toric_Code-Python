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

def make_svd_posi(u,v):
    if (np.sum(u.flatten()) < 0 and np.sum(u.flatten()) < 0):
        u *= -1
        v *= -1

def svd_4site(tensor, cutoff):
    dim_l = tensor.shape[0]
    dim_r = tensor.shape[-1]
    mat = np.reshape(tensor, (dim_l * 16, dim_r * 16))

    u,s,v = np.linalg.svd(mat)
    s = s[np.where(np.abs(s[:]) > cutoff)[0]]
    retain_dim = s.shape[0]
    mat_s = np.diag(s)
    u = np.dot(u[:, 0:retain_dim], np.sqrt(mat_s))
    v = np.dot(np.sqrt(mat_s), v[0:retain_dim, :])
    make_svd_posi(u, v)

    # do SVD for u
    u = np.reshape(u, (dim_l, 2, 2, 2, 2, retain_dim))
    u = np.einsum('iacbdn->iabcdn', u)
    u = np.reshape(u, (dim_l * 4, 4 * retain_dim))
    m1,su,m2 = np.linalg.svd(u)
    su = su[np.where(np.abs(su[:]) > cutoff)[0]]
    retain_dim_u = su.shape[0]
    mat_su = np.diag(su)
    m1 = np.dot(m1[:, 0:retain_dim_u], np.sqrt(mat_su))
    m1 = np.reshape(m1, (dim_l, 2, 2, retain_dim_u))
    m2 = np.dot(np.sqrt(mat_su), m2[0:retain_dim_u, :])
    m2 = np.reshape(m2, (retain_dim_u, 2, 2, retain_dim))
    make_svd_posi(m1, m2)

    # do SVD for v
    v = np.reshape(v, (retain_dim, 2, 2, 2, 2, dim_r))
    v = np.einsum('negfhm->nefghm', v)
    v = np.reshape(v, (4 * retain_dim, 4 * dim_r))
    m3,sv,m4 = np.linalg.svd(v)
    sv = sv[np.where(np.abs(sv[:]) > cutoff)[0]]
    retain_dim_v = sv.shape[0]
    mat_sv = np.diag(sv)
    m3 = np.dot(m3[:, 0:retain_dim_v], np.sqrt(mat_sv))
    m3 = np.reshape(m3, (retain_dim, 2, 2, retain_dim_v))
    m4 = np.dot(np.sqrt(mat_sv), m4[0:retain_dim_v, :])
    m4 = np.reshape(m4, (retain_dim_v, 2, 2, dim_r))
    make_svd_posi(m3, m4)

    return m1, m2, m3, m4

def svd_2site(tensor, cutoff):
    dim_l = tensor.shape[0]
    dim_r = tensor.shape[-1]
    mat = np.reshape(tensor, (dim_l * 4, dim_r * 4))

    u,s,v = np.linalg.svd(mat)
    s = s[np.where(np.abs(s[:]) > cutoff)[0]]
    retain_dim = s.shape[0]
    mat_s = np.diag(s)
    u = np.dot(u[:, 0:retain_dim], np.sqrt(mat_s))
    u = np.reshape(u, (dim_l, 2, 2, retain_dim))
    v = np.dot(np.sqrt(mat_s), v[0:retain_dim, :])
    v = np.reshape(v, (retain_dim, 2, 2, dim_r))
    make_svd_posi(u, v)

    return u, v

# set the orthogonality center of the MPO
def position(mpo, pos, cutoff):
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
        a = np.einsum('abij->iabj', a)
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
        b = np.einsum('ijab->iabj', b)
        mpo[i] = b
        # update mpo[i-1]
        mat_s = np.diag(s)
        u = np.dot(u[:, 0:retain_dim], mat_s)
        mpo[i-1] = np.einsum('iabj,js->iabs', mpo[i-1], u)

def gateTEvol(mpo, gateList, ttotal, tstep):
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
                ten_AA = np.einsum('ibek,kdgj,abcd,efgh->iafchj',mpo[sites[0]],mpo[sites[1]],gate.gate,gate2.gate)
                m1, m2 = svd_2site(ten_AA, 1.0E-10)
                mpo[sites[0]] = m1
                mpo[sites[1]] = m2
            # time evolution gate
            elif len(sites) == 4:
                gate2 = copy.copy(gate)
                gate2.gate = np.conj(gate.gate)
                position(mpo, sites[1], 1.0E-10)
                ten_AAAA = np.einsum('ibpk,kdrl,lfum,mhwj->ibpdrfuhwj',mpo[sites[0]],mpo[sites[1]],mpo[sites[2]],mpo[sites[3]])
                ten_AAAA = np.einsum('ibpdrfuhwj,abcdefgh,pqrsuvwx->iaqcsevgxj',ten_AAAA,gate.gate,gate2.gate)
                m1, m2, m3, m4 = svd_4site(ten_AAAA, 1.0E-10)
                mpo[sites[0]] = m1
                mpo[sites[1]] = m2
                mpo[sites[2]] = m3
                mpo[sites[3]] = m4
            # error handling
            else:
                print('Wrong number of sites')
                sys.exit()
            
            