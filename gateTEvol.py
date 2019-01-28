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

# set the orthogonality center of the MPO
def position(mpo, pos, cutoff):
    siteNum = len(mpo)

    # left normalization
    for i in range(pos):
        virDim = [mpo[i].shape[0], mpo[i].shape[1]]
        phyDim = [mpo[i].shape[2], mpo[i].shape[3]]
        # i,j: virtual leg; a,b: physical leg
        mat = np.einsum('iabj->abij', mpo[i])
        mat = np.reshape(mpo[i], (phyDim[0]*phyDim[1]*virDim[0], virDim[1]))
        a,s,v = np.linalg.svd(mat)
        s = s[np.where(np.abs(s[:]) > cutoff)[0]]
        retain_dim = s.shape[0]
        a = a[:, 0:retain_dim]
        v = v[0:retain_dim, :]
        # replace mpo[i] with a
        a = np.reshape(a, (phyDim[0],phyDim[1],virDim[0],retain_dim))
        a = np.einsum('abij->iabj', a)
        mpo[i] = a
        # update mpo[i+1]
        mpo[i+1] = np.einsum('s,si,', s, v, mpo[i+1])

    # right normalization
    for i in np.arange(siteNum-1, pos, -1, dtype=int):
        virDim = [mpo[i].shape[0], mpo[i].shape[1]]
        phyDim = [mpo[i].shape[2], mpo[i].shape[3]]
        # i,j: virtual leg; a,b: physical leg
        mat = np.einsum('iabj->ijab', mpo[i])
        mat = np.reshape(mpo[i], (virDim[0]*virDim[1]*phyDim[0], phyDim[1]))
        u,s,b = np.linalg.svd(mat)
        s = s[np.where(np.abs(s[:]) > cutoff)[0]]
        retain_dim = s.shape[0]
        b = b[0:retain_dim, :]
        # replace mpo[i] with b


def gateTEvol(mpo, gateList, ttotal, tstep):
    nt = int(ttotal/tstep + (1e-9 * (ttotal/tstep)))
    if (np.abs(nt*tstep-ttotal) > 1.0E-9): 
        print("Timestep not commensurate with total time")
        sys.exit()
    gateNum = len(gateList)

    for tt in range(nt):
        for g in range(gateNum):
            pass