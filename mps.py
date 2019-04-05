# 
#   mps.py
#   Toric_Code-Python
#   Operations related to MPS
#
#   created on Feb 7, 2019 by Yue Zhengyuan
#

import numpy as np
import scipy.linalg as LA
import sys
import copy
import para_dict as p
from itertools import product
# import atexit
# import line_profiler as lp

# profile = lp.LineProfiler()
# atexit.register(profile.print_stats)

def svd_truncate(u, s, v, args, normalize=True, rounding=False):
    """
    Truncate and round SVD results

    Returns
    ---------------
    Truncated u, s, v and
    The keeped number of singular values
    
    Parameters
    ---------------
    u, s, v : numpy array
        scipy.linalg.svd results
    args : dict
        parameters controlling SVD
    normalize : default True
        rescale the remaining singular value so that sum(s**2) is unchanged
    rounding : default False
        decide whether to round the SVD result matrices
    """
    # remove zero singular values
    s = s[np.where(s[:] > 0)[0]]
    old_s2 = np.dot(s, s)
    old_snorm = np.linalg.norm(s)
    trunc = 0
    origin_dim = s.shape[0]
    remove_dim = 0
    while trunc/old_s2 < args['cutoff']:
        remove_dim += 1
        trunc += s[-remove_dim] * s[-remove_dim]
    remove_dim -= 1
    retain_dim = min(args['bondm'], origin_dim - remove_dim)
    s = s[0:retain_dim]
    u = u[:, 0:retain_dim]
    v = v[0:retain_dim, :]
    # after truncating, scale the singular values so that
    # sum(s**2) is unchanged
    if normalize == True:
        new_snorm = np.linalg.norm(s)
        s *= old_snorm / new_snorm
    # rounding
    if rounding == True:
        u = np.around(u, decimals=10)
        s = np.around(s, decimals=10)
        v = np.around(v, decimals=10)
    return u, s, v, retain_dim

def position(psi, pos, args, oldcenter=-1, preserve_norm=True, compute_entg=False):
    """
    set the orthogonality center of the MPS |psi> to pos'th site

    Parameters
    ---------------
    oldcenter : int between -1 and len(psi) - 1 (default = -1)
        when old center <= 0,            do right canonization
        when old center == len(psi)-1,   do left canonization
    preserve_norm : default True 
        determine how s is combined with u,v in SVD
        Example: if going from left to right
            if True : u -> u * |s|; v -> s/|s| * v
                (norm of u is unchanged)
            if False: u -> u; v -> s * v
            (|s| = np.norm(s))
    compute_entg : default False
        if True: return the entanglement entropy between the two sides of the orthogonality center
    """
    phi = copy.copy(psi)
    siteNum = len(psi)
    # determine left/right canonization range
    if oldcenter == -1:               # initialize center to pos
        left = np.arange(0, pos, 1, dtype=int)
        right = np.arange(siteNum-1, pos, -1, dtype=int)
    elif (oldcenter >= siteNum or oldcenter < -1):
        sys.exit("Old center position out of range")
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
    for i in left:
        virDim = [phi[i].shape[0], phi[i].shape[-1]]
        phyDim = phi[i].shape[1]
        # i,j: virtual leg; a: physical leg
        mat = np.reshape(phi[i], (virDim[0]*phyDim, virDim[1]))
        a,s,v = LA.svd(mat, full_matrices=False, lapack_driver='gesvd')
        a,s,v,retain_dim = svd_truncate(a, s, v, args=args)
        # redistribute matrix norm
        if preserve_norm == True:
            s_norm = np.linalg.norm(s)
            a *= s_norm
            s /= s_norm
        # replace mps[i] with a
        a = np.reshape(a, (virDim[0],phyDim,retain_dim))
        phi[i] = a
        # update mps[i+1]
        mat_s = np.diag(s)
        v = np.dot(mat_s, v)
        phi[i+1] = np.einsum('si,iaj->saj', v, phi[i+1], optimize=True)
    # right canonization (siteNum-1 to pos+1)
    for i in right:
        virDim = [phi[i].shape[0], phi[i].shape[-1]]
        phyDim = phi[i].shape[1]
        # i,j: virtual leg; a: physical leg
        mat = np.reshape(phi[i], (virDim[0], phyDim*virDim[1]))
        u,s,b = LA.svd(mat, full_matrices=False, lapack_driver='gesvd')
        u,s,b,retain_dim = svd_truncate(u, s, b, args=args)
        # redistribute matrix norm
        if preserve_norm == True:
            s_norm = np.linalg.norm(s)
            b *= s_norm
            s /= s_norm
        # replace mps[i] with b
        b = np.reshape(b, (retain_dim,phyDim,virDim[1]))
        phi[i] = b
        # update mps[i-1]
        mat_s = np.diag(s)
        u = np.dot(u, mat_s)
        phi[i-1] = np.einsum('iaj,js->ias', phi[i-1], u, optimize=True)
    # compute entanglement entropy
    if compute_entg == True:
        # do svd for the matrix at the orthogonality center
        mat = np.reshape(phi[pos], (virDim[0]*phyDim, virDim[1]))
        s = LA.svd(mat, full_matrices=False, compute_uv=False, lapack_driver='gesvd')
        # normalize spectrum: sum(s**2) = 1
        s2 = s**2
        s2 = s2 / np.sum(s2)
        ln_s2 = np.log(s2)
        entg = -np.sum(s2 * ln_s2)
        return phi, entg
    else: 
        return phi

def normalize(psi, args):
    """
    Normalize MPS |psi> (and set orthogonality center at site 0)
    """
    phi = copy.copy(psi)
    pos = len(psi) - 1    # left canonical
    phi = position(phi, pos, args, preserve_norm=False)
    norm = np.tensordot(phi[pos], np.conj(phi[pos]), ([0,1,2],[0,1,2]))
    norm = np.sqrt(norm)
    phi[pos] /= norm
    return phi

def svd_nsite(n, tensor, dir, args, preserve_norm=True):
    """
    Do SVD to convert one large tensor into an n-site MPS
    
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
    if (len(tensor.shape) != n + 2):
        sys.exit('Wrong dimension of input tensor')
    virDim = [tensor.shape[0], tensor.shape[-1]]
    phyDim = list(tensor.shape[1:-1])
    result = []
    mat = copy.copy(tensor)
    
    if dir == 'Fromleft':
        old_retain_dim = virDim[0]
        for i in np.arange(0, n - 1, 1, dtype=int):
            mat = np.reshape(mat, (old_retain_dim * phyDim[i], 
            virDim[1] * np.prod(phyDim[i+1 : len(phyDim)])))
            u,s,v = LA.svd(mat, full_matrices=False, lapack_driver='gesvd')
            u,s,v,new_retain_dim = svd_truncate(u, s, v, args=args)
            # redistribute matrix norm
            if preserve_norm == True:
                scale = np.linalg.norm(s) ** (1 / (n - i))
                u *= scale
                s /= scale
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
            u,s,v = LA.svd(mat, full_matrices=False, lapack_driver='gesvd')
            u,s,v,new_retain_dim = svd_truncate(u, s, v, args=args)
            # redistribute matrix norm
            if preserve_norm == True:
                scale = np.linalg.norm(s) ** (1 / (i + 1))
                v *= scale
                s /= scale
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

def applyMPOtoMPS(op, psi, args):
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

def gateTEvol(psi, gateList, ttotal, tstep, args):
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
    # periodic = args['xperiodic']
    phi = position(phi, gateList[0].sites[0], args=args)
    oldcenter = 0
    # if the acting range of the gate crosses the boundary
    # the orthogonality center of the MPS should be reset
    for tt, g in product(range(nt), range(gateNum)):
        gate = gateList[g].gate
        sites = gateList[g].sites
        if len(sites) == 2:
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
            ten_AA = np.einsum('ibk,kdj,abcd->iacj',
            phi[sites[0]], phi[sites[1]], gate, optimize=True)
            # do svd to restore 2 sites
            if g < gateNum - 1:
                if gateList[g+1].sites[0] >= gateList[g].sites[-1]:
                    result = svd_nsite(2, ten_AA, 'Fromleft', args=args)
                    for i in range(2):
                        phi[sites[i]] = result[i]
                    # gate crosses boundary
                    if gateList[g].sites[-1] < gateList[g].sites[0]:
                        oldcenter = -1 # reset orthogonality center
                    else:
                        oldcenter = sites[-1]
                    phi = position(phi, gateList[g+1].sites[0], oldcenter=oldcenter, args=args)
                    oldcenter = gateList[g+1].sites[0]
                else:
                    result = svd_nsite(2, ten_AA, args=args, dir='Fromright')
                    for i in range(2):
                        phi[sites[i]] = result[i]
                    # gate crosses boundary
                    if gateList[g].sites[-1] < gateList[g].sites[0]:
                        oldcenter = -1 # reset orthogonality center
                    else:
                        oldcenter = sites[0]
                    phi = position(phi, gateList[g+1].sites[-1], oldcenter=oldcenter, args=args)
                    oldcenter = gateList[g+1].sites[-1]
            else:
                result = svd_nsite(2, ten_AA, 'Fromright', args=args)
                for i in range(2):
                    phi[sites[i]] = result[i]
                # gate crosses boundary
                if gateList[g].sites[-1] < gateList[g].sites[0]:
                    oldcenter = -1 # reset orthogonality center
                else:
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
            ten_AAAA = np.einsum('ibj,jdk,kfl,lhm->ibdfhm',
            phi[sites[0]],phi[sites[1]],phi[sites[2]],phi[sites[3]], optimize=True)
            ten_AAAA = np.einsum('ibdfhm,abcdefgh->iacegm',
            ten_AAAA, gate, optimize=True)
            # do svd to restore 4 sites
            if g < gateNum:
                if gateList[g+1].sites[0] >= gateList[g].sites[-1]:
                    result = svd_nsite(4, ten_AAAA, 'Fromleft', args=args)
                    for i in range(4):
                        phi[sites[i]] = result[i]
                    # gate crosses boundary
                    if gateList[g].sites[-1] < gateList[g].sites[0]:
                        oldcenter = -1 # reset orthogonality center
                    else:
                        oldcenter = sites[-1]
                    phi = position(phi, gateList[g+1].sites[0], oldcenter=oldcenter, args=args)
                    oldcenter = gateList[g+1].sites[0]
                else:
                    result = svd_nsite(4, ten_AAAA, 'Fromright',args=args)
                    for i in range(4):
                        phi[sites[i]] = result[i]
                    # gate crosses boundary
                    if gateList[g].sites[-1] < gateList[g].sites[0]:
                        oldcenter = -1 # reset orthogonality center
                    else:
                        oldcenter = sites[0]
                    phi = position(phi, gateList[g+1].sites[-1], oldcenter=oldcenter, args=args)
                    oldcenter = gateList[g+1].sites[-1]
            else:
                result = svd_nsite(4, ten_AAAA, 'Fromright', args=args)
                for i in range(4):
                    phi[sites[i]] = result[i]
                # gate crosses boundary
                if gateList[g].sites[-1] < gateList[g].sites[0]:
                    oldcenter = -1 # reset orthogonality center
                else:
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

def sum(psi1, psi2, args, compress="svd"):
    """
    Sum two MPS's

    Parameters
    ------------
    psi1, psi2 : list of numpy arrays
        The two MPSs to be added
    compress : None / "var" / "svd"
        compression method
    args : dict
        parameters controlling SVD
    """
    periodic = args['xperiodic']
    # dimension check
    if len(psi1) != len(psi2):
        sys.exit("The lengths of the two MPSs do not match")
    for i in range(len(psi1)):
        if psi1[i].shape[1] != psi2[i].shape[1]:
            sys.exit("The physical dimensions of the two MPSs do not match")
    result = []

    # special treatment is needed at the first and the last site
    # for open boundary condition
    if (compress == None or compress == "svd"):
        # first site
        i = 0
        virDim1 = [psi1[i].shape[0], psi1[i].shape[-1]]
        virDim2 = [psi2[i].shape[0], psi2[i].shape[-1]]
        phyDim = psi1[i].shape[1]
        if periodic == False:
            virDim = [virDim1[0], virDim1[1]+virDim2[1]]
            result.append(np.zeros((virDim[0], phyDim, virDim[1]), dtype=complex))
            # make row "vector"
            result[i][0:virDim1[0],:,0:virDim1[1]] = psi1[i]
            result[i][0:virDim1[0],:,virDim1[1]:virDim[1]] = psi2[i]
        elif periodic == True:
            virDim = [virDim1[0]+virDim2[0], virDim1[1]+virDim2[1]]
            result.append(np.zeros((virDim[0], phyDim, virDim[1]), dtype=complex))
            # direct sum
            result[i][0:virDim1[0],:,0:virDim1[1]] = psi1[i]
            result[i][virDim1[0]:virDim[0],:,virDim1[1]:virDim[1]] = psi2[i]

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
        if periodic == False:
            virDim = [virDim1[0]+virDim2[0], virDim1[1]]
            result.append(np.zeros((virDim[0], phyDim, virDim[1]), dtype=complex))
            # make column "vector"
            result[i][0:virDim1[0],:,0:virDim1[1]] = psi1[i]
            result[i][virDim1[0]:virDim[0],:,0:virDim1[1]] = psi2[i]
        elif periodic == True:
            virDim = [virDim1[0]+virDim2[0], virDim1[1]+virDim2[1]]
            result.append(np.zeros((virDim[0], phyDim, virDim[1]), dtype=complex))
            # direct sum
            result[i][0:virDim1[0],:,0:virDim1[1]] = psi1[i]
            result[i][virDim1[0]:virDim[0],:,virDim1[1]:virDim[1]] = psi2[i]

        # remove useless physical legs (will add later) 
        if compress == "svd":
            result = position(result, 0, args=args)

    # variational optimization
    elif (compress == "var"):
        pass

    else: 
        sys.exit("Wrong compress parameter")
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
            cond = np.where(np.abs(psi[i]) > 1.0E-10)
            index = np.transpose(cond)
            elem = psi[i][cond]
            for j in range(len(index)):
                for leg in range(legNum):
                    f.write(str(index[j, leg]) + '\t')
                f.write(str(elem[j]) + '\n')
            # separation line
            f.write("--------------------\n")
    