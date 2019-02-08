# 
#   mps.py
#   Toric_Code-Python
#   Operations related to MPS
#
#   created on Feb 7, 2019 by Yue Zhengyuan
#

import numpy as np
import sys
import copy

def svd_truncate(u, s, v, cutoff, bondm):
    """
    Truncate SVD results

    Returns
    ---------------
    Truncated u, s, v
    The keeped number of singular values
    
    Parameters
    ---------------
    u, s, v : numpy array
        np.linalg.svd results
    cutoff : float
        largest value of s_max/s_min
    bondm : int
        largest virtual bond dimension allowed
    """
    # remove zero sigular values
    s = s[np.where(s[:] > 0)[0]]
    s_max = max(s)
    retain_dim = min(bondm, len(np.where(np.abs(s_max/s[:]) < cutoff)[0]))
    s = s[0:retain_dim]
    u = u[:, 0:retain_dim]
    v = v[0:retain_dim, :]
    return u, s, v, retain_dim

def position(mps, pos, cutoff, bondm):
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
    bondm : int
        largest virtual bond dimension allowed
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
        a,s,v,retain_dim = svd_truncate(a, s, v, cutoff, bondm)
        # replace mps[i] with a
        a = np.reshape(a, (phyDim,virDim[0],retain_dim))
        a = np.einsum('ais->ias', a)
        mps[i] = a
        # update mps[i+1]
        mat_s = np.diag(s)
        v = np.dot(mat_s, v)
        mps[i+1] = np.einsum('si,iaj->saj', v, mps[i+1])

    # right normalization
    for i in np.arange(siteNum-1, pos, -1, dtype=int):
        virDim = [mps[i].shape[0], mps[i].shape[-1]]
        phyDim = mps[i].shape[1]
        # i,j: virtual leg; a: physical leg
        mat = np.einsum('iaj->ija', mps[i])
        mat = np.reshape(mps[i], (virDim[0], virDim[1]*phyDim))
        u,s,b = np.linalg.svd(mat)
        u,s,b,retain_dim = svd_truncate(u, s, b, cutoff, bondm)
        # replace mps[i] with b
        b = np.reshape(b, (retain_dim,virDim[1],phyDim))
        b = np.einsum('sja->saj', b)
        mps[i] = b
        # update mps[i-1]
        mat_s = np.diag(s)
        u = np.dot(u, mat_s)
        mps[i-1] = np.einsum('iaj,js->ias', mps[i-1], u)
    
def normalize(psi, cutoff, bondm):
    """
    Normalize MPS

    Parameters
    ---------------
    psi : list of numpy arrays
        MPS to be normalized
    cutoff : float
        largest value of s_max/s_min
    bondm : int
        largest virtual bond dimension allowed
    """
    phi = copy.copy(psi)
    center_pos = int(len(phi)/2)
    position(phi, center_pos, cutoff, bondm)
    norm = np.tensordot(phi[center_pos], np.conj(phi[center_pos]), ([0,1,2],[0,1,2]))
    norm = np.sqrt(norm)
    phi /= norm
    phi = list(phi)
    return phi

def applyMPOtoMPS(mpo, mps, cutoff, bondm):
    """
    Multiply MPO and MPS
    """
    siteNum = len(mpo)
    if (len(mps) != siteNum):
        print('Number of sites of the two MPOs do not match.')
        sys.exit()
    # dimension check
    for site in range(siteNum):
        if mpo[site].shape[2] != mps[site].shape[1]:
            print('Physical leg dimensions of the MPO and the MPS do not match.')
            sys.exit()
    # contraction
    result = []
    for site in range(siteNum):
        group = np.einsum('iabj,kbl->ikajl',mpo[site],mps[site])
        shape = group.shape
        group = np.reshape(group, (shape[0]*shape[1], shape[2], shape[3]*shape[4]))
        result.append(group)
    # restore MPO form by SVD and truncate virtual links
    # set orthogonality center at the middle of the MPO
    position(result, int(siteNum/2), cutoff, bondm)
    result = normalize(result, cutoff, bondm)
    return result

def overlap(mps1, mps2, cutoff, bondm):
    """
    Calculate the inner product of two MPS's

    Parameters
    ------------
    mps1 : list of numpy arrays
        The MPS above (will be complex-conjugated)
    mps2 : list of numpy arrays
        The MPS below
    """
    siteNum = len(mps1)
    if (len(mps2) != siteNum):
        print('Number of sites of the two MPOs do not match.')
        sys.exit()
    # dimension check
    for site in range(siteNum):
        if mps1[site].shape[1] != mps2[site].shape[1]:
            print('Physical leg dimensions of the two MPOs do not match.')
            sys.exit()
    # contraction
    result = []
    for site in range(siteNum):
        group = np.einsum('iaj,kal->ikjl',np.conj(mps1[site]),mps2[site])
        shape = group.shape
        group = np.reshape(group, (shape[0]*shape[1],1,shape[2]*shape[3]))
        result.append(group)
    # restore MPO form by SVD and truncate virtual links
    # set orthogonality center at the middle of the MPO
    position(result, int(siteNum/2), cutoff, bondm)
    return result
