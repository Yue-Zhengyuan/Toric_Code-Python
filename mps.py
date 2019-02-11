# 
#   py
#   Toric_Code-Python
#   Operations related to MPS
#
#   created on Feb 7, 2019 by Yue Zhengyuan
#

import numpy as np
import sys
import copy
import gates

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

def position(psi, pos, cutoff, bondm):
    """
    set the orthogonality center of the MPS
    
    Parameters
    ----------------
    psi : list of numpy arrays
        the MPS to be acted on
    pos : int
        the position of the orthogonality center
    cutoff: float
        largest value of s_max/s_min
    bondm : int
        largest virtual bond dimension allowed
    """
    phi = copy.copy(psi)
    siteNum = len(phi)
    # left normalization
    for i in range(pos):
        virDim = [phi[i].shape[0], phi[i].shape[-1]]
        phyDim = phi[i].shape[1]
        # i,j: virtual leg; a: physical leg
        mat = np.einsum('iaj->aij', phi[i])
        mat = np.reshape(phi[i], (phyDim*virDim[0], virDim[1]))
        a,s,v = np.linalg.svd(mat)
        a,s,v,retain_dim = svd_truncate(a, s, v, cutoff, bondm)
        # replace mps[i] with a
        a = np.reshape(a, (phyDim,virDim[0],retain_dim))
        a = np.einsum('ais->ias', a)
        phi[i] = a
        # update mps[i+1]
        mat_s = np.diag(s)
        v = np.dot(mat_s, v)
        phi[i+1] = np.einsum('si,iaj->saj', v, phi[i+1])

    # right normalization
    for i in np.arange(siteNum-1, pos, -1, dtype=int):
        virDim = [phi[i].shape[0], phi[i].shape[-1]]
        phyDim = phi[i].shape[1]
        # i,j: virtual leg; a: physical leg
        mat = np.einsum('iaj->ija', phi[i])
        mat = np.reshape(phi[i], (virDim[0], virDim[1]*phyDim))
        u,s,b = np.linalg.svd(mat)
        u,s,b,retain_dim = svd_truncate(u, s, b, cutoff, bondm)
        # replace mps[i] with b
        b = np.reshape(b, (retain_dim,virDim[1],phyDim))
        b = np.einsum('sja->saj', b)
        phi[i] = b
        # update mps[i-1]
        mat_s = np.diag(s)
        u = np.dot(u, mat_s)
        phi[i-1] = np.einsum('iaj,js->ias', phi[i-1], u)
    return phi

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
    u,s,v,retain_dim = svd_truncate(u, s, v, cutoff, bondm)
    mat_s = np.diag(s)
    u = np.dot(u, np.sqrt(mat_s))
    v = np.dot(np.sqrt(mat_s), v)
    u = np.reshape(u, (virDim[0], phyDim, retain_dim))
    v = np.reshape(v, (retain_dim, phyDim, virDim[1]))
    return u, v

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
                phi = position(phi, sites[0], cutoff, bondm)
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
                phi = position(phi, sites[1], cutoff, bondm)
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
                phi = position(phi, sites[0], cutoff, bondm)
                phi[sites[0]] = np.reshape(phi[sites[0]], (phi[sites[0]].shape[0],2,2,phi[sites[0]].shape[-1]))
                m1, m2 = svd_2site(phi[sites[0]], cutoff, bondm)
                phi[sites[0]] = m1
                phi.insert(sites[1], m2)

                phi = position(phi, sites[2], cutoff, bondm)
                phi[sites[2]] = np.reshape(phi[sites[2]], (phi[sites[2]].shape[0],2,2,phi[sites[2]].shape[-1]))
                m3, m4 = svd_2site(phi[sites[2]], cutoff, bondm)
                phi[sites[2]] = m3
                phi.insert(sites[3], m4)
            # error handling
            else:
                print('Wrong number of sites')
                sys.exit()
    # return a guaged and normalized MPS |phi>
    phi = normalize(phi, cutoff, bondm)
    return phi
