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
import gates
from itertools import product

def svd_truncate(u, s, v, args={'cutoff':1.0E-5, 'bondm':200, 'scale':False}):
    """
    Truncate SVD results

    Returns
    ---------------
    Truncated u, s, v and
    The keeped number of singular values
    
    Parameters
    ---------------
    u, s, v : numpy array
        np.linalg.svd results
    args : dict
        parameters controlling SVD
    """
    # get arguments
    cutoff = args.setdefault('cutoff', 1.0E-5)
    bondm = args.setdefault('bondm', 200)
    scale = args.setdefault('scale', False)
    # remove zero sigular values
    s = s[np.where(s[:] > 0)[0]]
    s_sum = np.linalg.norm(s)**2
    trunc = 0
    origin_dim = s.shape[0]
    remove_dim = 0
    while trunc/s_sum < cutoff:
        remove_dim += 1
        trunc += s[-remove_dim]**2
    remove_dim -= 1
    retain_dim = min(bondm, origin_dim - remove_dim)
    s = s[0:retain_dim]
    u = u[:, 0:retain_dim]
    v = v[0:retain_dim, :]
    if scale == True:
        average = np.average(s)
        s /= average
    return u, s, v, retain_dim

def position(psi, pos, oldcenter=-1, args={'cutoff':1.0E-5, 'bondm':200, 'scale':False}):
    """
    set the orthogonality center of the MPS |psi> to pos'th site

    Parameters
    ---------------
    old center : int (default = -1)
        when old center < 0,            do right canonization
        when old center > len(psi)-1,   do left canonization
    """
    phi = copy.copy(psi)
    siteNum = len(phi)
    if oldcenter < 0:               # initialize center to pos
        left = np.arange(0, pos, 1, dtype=int)
        right = np.arange(siteNum-1, pos, -1, dtype=int)
    elif oldcenter >= siteNum:
        sys.exit("Old gauge center out of range")
    else:                           # set center to pos
        if pos == oldcenter: # no need to do canonization at all
            left = []
            right = []
        elif pos > oldcenter:
            left = np.arange(oldcenter, pos, 1, dtype=int)
            right = [] # no need to do right canonization
        elif pos < oldcenter:
            left = []
            right = np.arange(oldcenter, pos, -1, dtype=int)
        else:
            sys.exit("Gauge center out of range")
    # left canonization (0 to pos - 1)
    # for i in range(pos):
    for i in left:
        virDim = [phi[i].shape[0], phi[i].shape[-1]]
        phyDim = phi[i].shape[1]
        # i,j: virtual leg; a: physical leg
        mat = np.reshape(phi[i], (virDim[0]*phyDim, virDim[1]))
        a,s,v = np.linalg.svd(mat)
        a,s,v,retain_dim = svd_truncate(a, s, v, args=args)
        # replace mps[i] with a
        a = np.reshape(a, (virDim[0],phyDim,retain_dim))
        phi[i] = a
        # update mps[i+1]
        mat_s = np.diag(s)
        v = np.dot(mat_s, v)
        phi[i+1] = np.einsum('si,iaj->saj', v, phi[i+1], optimize=True)
        # phi[i], phi[i+1] = signCorrect(phi[i], phi[i+1])
    # right canonization (siteNum-1 to pos+1)
    # for i in np.arange(siteNum-1, pos, -1, dtype=int):
    for i in right:
        virDim = [phi[i].shape[0], phi[i].shape[-1]]
        phyDim = phi[i].shape[1]
        # i,j: virtual leg; a: physical leg
        mat = np.reshape(phi[i], (virDim[0], phyDim*virDim[1]))
        u,s,b = np.linalg.svd(mat)
        u,s,b,retain_dim = svd_truncate(u, s, b, args=args)
        # replace mps[i] with b
        b = np.reshape(b, (retain_dim,phyDim,virDim[1]))
        phi[i] = b
        # update mps[i-1]
        mat_s = np.diag(s)
        u = np.dot(u, mat_s)
        phi[i-1] = np.einsum('iaj,js->ias', phi[i-1], u, optimize=True)
        # phi[i-1], phi[i] = signCorrect(phi[i-1], phi[i])
    return phi

def normalize(psi, args={'cutoff':1.0E-5, 'bondm':200, 'scale':False}):
    """
    Normalize MPS |psi> (and set orthogonality center at site 0)
    """
    phi = copy.copy(psi)
    pos = 0     # right canonical
    phi = position(phi, pos, args=args)
    norm = np.tensordot(phi[pos], np.conj(phi[pos]), ([0,1,2],[0,1,2]))
    norm = np.sqrt(norm)
    phi[pos] /= norm
    return phi

def svd_nsite(n, tensor, dir, args={'cutoff':1.0E-5, 'bondm':200, 'scale':False}):
    """
    Do SVD to decomposite one large tensor into n site tensors of an MPS
    (u, v are made "positive" in terms of their eigenvalue)
    
    Parameters
    ---------------
    n : int
        number of site tensors to be produces
    tensor : numpy array
        the 2-site tensor to be decomposed
    dir: 'Fromleft'/'Fromright'
        if dir == 'left'/'right', the last/first of the n sites will be orthogonality center of the n tensors
    args : dict
        parameters controlling SVD
    """
    if (len(tensor.shape) != n + 2):
        sys.exit('Wrong dimension of input tensor')
    virDim = [tensor.shape[0], tensor.shape[-1]]
    phyDim = list(tensor.shape[1:-1])
    result = []
    mat = copy.copy(tensor)
    # rounding
    mat = np.around(mat, decimals=8)
    if dir == 'Fromleft':
        old_retain_dim = virDim[0]
        for i in np.arange(0, n - 1, 1, dtype=int):
            mat = np.reshape(mat, (old_retain_dim * phyDim[i], 
            virDim[1] * np.prod(phyDim[i+1 : len(phyDim)])))
            u,s,v = np.linalg.svd(mat, full_matrices=False)
            u,s,v,new_retain_dim = svd_truncate(u, s, v, args=args)
            mat_s = np.diag(s)
            v = np.dot(mat_s, v)
            u = np.reshape(u, (old_retain_dim, phyDim[i], new_retain_dim))
            result.append(u)
            mat = copy.copy(v)
            old_retain_dim = new_retain_dim
        v = np.reshape(v, (old_retain_dim, phyDim[-1], virDim[-1]))
        result.append(v)
    if dir == 'Fromright':
        old_retain_dim = virDim[1]
        for i in np.arange(n - 1, 0, -1, dtype=int):
            mat = np.reshape(mat, (virDim[0] * np.prod(phyDim[0 : i]),
            phyDim[i] * old_retain_dim))
            u,s,v = np.linalg.svd(mat, full_matrices=False)
            u,s,v,new_retain_dim = svd_truncate(u, s, v, args=args)
            mat_s = np.diag(s)
            u = np.dot(u, mat_s)
            v = np.reshape(v, (new_retain_dim, phyDim[i], old_retain_dim))
            result.append(v)
            mat = copy.copy(u)
            old_retain_dim = new_retain_dim
        u = np.reshape(u, (virDim[0], phyDim[0], old_retain_dim))
        result.append(u)
        result.reverse()
    return result

def applyMPOtoMPS(op, psi, args={'cutoff':1.0E-5, 'bondm':200, 'scale':False}):
    """
    Multiply MPO and MPS: op|psi>
    """
    siteNum = len(op)
    if (len(psi) != siteNum):
        print('Number of sites of the two MPOs do not match.')
        sys.exit()
    # dimension check
    for site in range(siteNum):
        if op[site].shape[2] != psi[site].shape[1]:
            print('Physical leg dimensions of the MPO and the MPS do not match.')
            sys.exit()
    # contraction
    result = []
    for site in range(siteNum):
        group = np.einsum('iabj,kbl->ikajl',op[site],psi[site], optimize=True)
        shape = group.shape
        group = np.reshape(group, (shape[0]*shape[1], shape[2], shape[3]*shape[4]))
        result.append(group)
    # restore MPO form by SVD and truncate virtual links
    # set orthogonality center at the middle of the MPO
    result = normalize(result, args=args)
    return result

def overlap(psi1, psi2):
    """
    Calculate the inner product <psi1|psi2>
    """
    siteNum = len(psi1)
    if (len(psi2) != siteNum):
        print('Number of sites of the two MPOs do not match.')
        sys.exit()
    # dimension check
    for site in range(siteNum):
        if psi1[site].shape[1] != psi2[site].shape[1]:
            print('Physical leg dimensions of the two MPOs do not match.')
            sys.exit()
    # partial contraction
    #      ___
    #  i --| |-- j
    #      -|-
    #       a
    #      _|_
    #  k --| |-- l
    #      --- 
    # 
    result = []
    for site in range(siteNum):
        group = np.einsum('iaj,kal->ikjl',np.conj(psi1[site]),psi2[site], optimize=True)
        result.append(group)
    # restore MPO form by SVD and truncate virtual links
    # set orthogonality center
    # full contraction
    elem = result[0]
    for site in np.arange(1, siteNum, 1, dtype=int):
        elem = np.tensordot(elem, result[site], axes=2)
    elem = np.reshape(elem, 1)
    return elem[0]

def matElem(psi1, op, psi2):
    """
    Calculate matrix element <psi1|op|psi2>
    """
    # site number check
    if (len(psi1) != len(op) or len(psi2) != len(op)):
        print('Number of sites of MPS and MPO do not match.')
        sys.exit()
    # dimension check
    siteNum = len(op)
    for site in range(siteNum):
        if (psi1[site].shape[1] != op[site].shape[1] or psi2[site].shape[1] != op[site].shape[2]):
            print('Physical leg dimensions of MPO and MPS do not match.')
            sys.exit()
    # partial contraction
    #      ___
    #  i --| |-- j
    #      -|-
    #       a
    #      _|_
    #  k --| |-- l
    #      -|- 
    #       b
    #      _|_
    #  m --| |-- n
    #      ---
    # 
    result = []
    for site in range(siteNum):
        group = np.einsum('iaj,kabl,mbn->ikmjln', 
        np.conj(psi1[site]), op[site], psi2[site], optimize=True)
        result.append(group)
    # full contraction
    elem = result[0]
    for site in np.arange(1, siteNum, 1, dtype=int):
        elem = np.tensordot(elem, result[site], axes=3)
    elem = np.reshape(elem, 1)
    return elem[0]

def gateTEvol(psi, gateList, ttotal, tstep, 
args={'cutoff':1.0E-5, 'bondm':200, 'scale':False}):
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
    args : dict
        parameters controlling SVD
    """
    phi = copy.copy(psi)
    nt = int(ttotal/tstep + (1e-9 * (ttotal/tstep)))
    if (np.abs(nt*tstep-ttotal) > 1.0E-9): 
        sys.exit("Timestep not commensurate with total time")
    gateNum = len(gateList)

    phi = position(phi, gateList[0].sites[0], args=args)
    oldcenter = 0
    for tt in range(nt):
        for g in range(gateNum):
            gate = gateList[g].gate
            sites = gateList[g].sites
            if len(sites) == 1:
                # contraction
                #
                #       a
                #      _|_
                #      | |      gate = exp(-i H dt)
                #      -|-
                #       b
                #      _|_
                #  i --| |-- k
                #      --- 
                #
                ten_AA = np.einsum('ibk,ab->iak',phi[sites[0]],gate, optimize=True)
                phi[sites[0]] = ten_AA
                if g < gateNum - 1:
                    phi = position(phi, gateList[g+1].sites[0], oldcenter=oldcenter, args=args)
                    oldcenter = gateList[g+1].sites[0]
                else:
                    phi = position(phi, 0, oldcenter=oldcenter, args=args)
                    oldcenter = 0
            elif len(sites) == 2:
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
                ten_AA = np.einsum('ibk,kdj,abcd->iacj',phi[sites[0]],phi[sites[1]],gate, optimize=True)
                # do svd to restore 2 sites
                if g < gateNum - 1:
                    if gateList[g+1].sites[0] >= gateList[g].sites[-1]:
                        result = svd_nsite(2, ten_AA, 'Fromleft', args=args)
                        for i in range(2):
                            phi[sites[i]] = result[i]
                        oldcenter = sites[-1]
                        phi = position(phi, gateList[g+1].sites[0], oldcenter=oldcenter, args=args)
                        oldcenter = gateList[g+1].sites[0]
                    else:
                        result = svd_nsite(2, ten_AA, args=args, dir='Fromright')
                        for i in range(2):
                            phi[sites[i]] = result[i]
                        oldcenter = sites[0]
                        phi = position(phi, gateList[g+1].sites[-1], oldcenter=oldcenter, args=args)
                        oldcenter = gateList[g+1].sites[-1]
                else:
                    result = svd_nsite(2, ten_AA, 'Fromright', args=args)
                    for i in range(2):
                        phi[sites[i]] = result[i]
                    oldcenter = sites[0]
                    phi = position(phi, 0, oldcenter=oldcenter, args=args)
                    oldcenter = 0
            elif len(sites) == 4:
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
                ten_AAAA = np.einsum('ibj,jdk,kfl,lhm->ibdfhm',phi[sites[0]],phi[sites[1]],phi[sites[2]],phi[sites[3]], optimize=True)
                ten_AAAA = np.einsum('ibdfhm,abcdefgh->iacegm',ten_AAAA,gate, optimize=True)
                # do svd to restore 4 sites
                if g < gateNum:
                    if gateList[g+1].sites[0] >= gateList[g].sites[-1]:
                        result = svd_nsite(4, ten_AAAA, 'Fromleft', args=args)
                        for i in range(4):
                            phi[sites[i]] = result[i]
                        oldcenter = sites[-1]
                        phi = position(phi, gateList[g+1].sites[0], oldcenter=oldcenter, args=args)
                        oldcenter = gateList[g+1].sites[0]
                    else:
                        result = svd_nsite(4, ten_AAAA, 'Fromright',args=args)
                        for i in range(4):
                            phi[sites[i]] = result[i]
                        oldcenter = sites[0]
                        phi = position(phi, gateList[g+1].sites[-1], oldcenter=oldcenter, args=args)
                        oldcenter = gateList[g+1].sites[-1]
                else:
                    result = svd_nsite(4, ten_AAAA, 'Fromright', args=args)
                    for i in range(4):
                        phi[sites[i]] = result[i]
                    oldcenter = sites[0]
                    phi = position(phi, 0, oldcenter=oldcenter, args=args)
                    oldcenter = 0
            # error handling
            else:
                print('Wrong number of sites')
                sys.exit()
    # return a guaged and normalized MPS |phi>
    phi = normalize(phi, args=args)
    return phi

def sum(psi1, psi2, args={'cutoff':1.0E-5, 'bondm':200, 'scale':False}):
    """
    Sum two MPS's

    Parameters
    ------------
    psi1, psi2 : list of numpy arrays
        The two MPSs to be added
    args : dict
        parameters controlling SVD
    """
    # dimension check
    if len(psi1) != len(psi2):
        sys.exit("The lengths of the two MPSs do not match")
    for i in range(len(psi1)):
        if psi1[i].shape[1] != psi2[i].shape[1]:
            sys.exit("The physical dimensions of the two MPSs do not match")
    result = []

    # special treatment is needed at the first and the last site
    # for open boundary condition

    # first site
    i = 0
    virDim1 = [psi1[i].shape[0], psi1[i].shape[-1]]
    virDim2 = [psi2[i].shape[0], psi2[i].shape[-1]]
    phyDim = psi1[i].shape[1]
    virDim = [virDim1[0], virDim1[1]+virDim2[1]]
    result.append(np.zeros((virDim[0], phyDim, virDim[1]), dtype=complex))
    # make row "vector"
    result[i][0:virDim1[0],:,0:virDim1[1]] = psi1[i]
    result[i][0:virDim1[0],:,virDim1[1]:virDim[1]] = psi2[i]

    # other sites
    for i in np.arange(1, len(psi1)-1, 1, dtype=int):
        virDim1 = [psi1[i].shape[0], psi1[i].shape[-1]]
        virDim2 = [psi2[i].shape[0], psi2[i].shape[-1]]
        phyDim = psi1[i].shape[1]
        virDim = [virDim1[0]+virDim2[0], virDim1[1]+virDim2[1]]
        result.append(np.zeros((virDim[0], phyDim, virDim[1]), dtype=complex))
        # direct sum
        result[i][0:virDim1[0],:,0:virDim1[1]] = psi1[i]
        result[i][virDim1[0]:virDim[0],:,virDim1[1]:virDim[1]] = psi2[i]

    # last site
    i = len(psi1) - 1
    virDim1 = [psi1[i].shape[0], psi1[i].shape[-1]]
    virDim2 = [psi2[i].shape[0], psi2[i].shape[-1]]
    phyDim = psi1[i].shape[1]
    virDim = [virDim1[0]+virDim2[0], virDim1[1]]
    result.append(np.zeros((virDim[0], phyDim, virDim[1]), dtype=complex))
    # make column "vector"
    result[i][0:virDim1[0],:,0:virDim1[1]] = psi1[i]
    result[i][virDim1[0]:virDim[0],:,0:virDim1[1]] = psi2[i]

    # remove useless physical legs (will add later) 

    result = position(result, 0, args=args)
    return result

def printdata(psi):
    """
    Print MPS to screen
    """
    for i in range(len(psi)):
        print('\nSite',i,', Shape',psi[i].shape)
        cond = np.where(np.abs(psi[i]) > 1.0E-8)
        index = np.transpose(cond)
        elem = psi[i][cond]
        for j in range(len(index)):
            print(index[j], elem[j])

def save_to_file(psi, filename):
    """
    Save MPS (shape and nonzero elements) to (txt) file

    Parameters
    ----------
    op : list of numpy arrays
        the MPO to be written to file
    filename : string
        name of the output file
    """
    with open(filename, 'w+') as f:
        for i in range(len(psi)):
            legNum = len(psi[i].shape)
            f.write(str(i) + '\t')
            for leg in range(legNum):
                if leg != legNum - 1:
                    f.write(str(psi[i].shape[leg]) + '\t')
                else:
                    f.write(str(psi[i].shape[leg]) + '\n')
            cond = np.where(np.abs(psi[i]) > 1.0E-8)
            index = np.transpose(cond)
            elem = psi[i][cond]
            for j in range(len(index)):
                for leg in range(legNum):
                    f.write(str(index[j, leg]) + '\t')
                f.write(str(elem[j]) + '\n')
            # separation line
            f.write("--------------------\n")
    