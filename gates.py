# 
#   gates.py
#   Toric_Code-Python
#   generate time-evolution gates and swap gates
#
#   created on Jan 21, 2019 by Yue Zhengyuan
#   code influcenced by finite_TMPS - trotter.h
#                   and iTensor - bondgate.h
#

import numpy as np
import sys
from itertools import product
import para_dict as p
import copy

# index order convention
# 
# Tensor
#       a      c
#      _|_    _|_
#  i --| |----| |--j
#      -|-    -|-
#       b      d
#
#  index order: iabcdj
# 
# Gate
#       a      c
#      _|_    _|_
#      | |----| |
#      -|-    -|-
#       b      d
#
#  index order: abcd

# exp(x) = 1 + x +  x^2/2! + x^3/3! ..
# = 1 + x * (1 + x/2 *(1 + x/3 * (...
# ~ ((x/3 + 1) * x/2 + 1) * x + 1

# 4-site time-evolution gate
def toExpH4(ham, order):
    term = copy.copy(ham)
    unit = np.einsum('ab,cd,ef,gh->abcdefgh', p.iden, p.iden, p.iden, p.iden)
    for i in np.arange(order, 0, -1, dtype=int):
        term /= i
        gate = unit + term
        term = np.einsum('abcdefgh,bwdxfyhz->awcxeygz', gate, ham)
    return gate

# 1-site time-evolution gate
def toExpH1(ham, order):
    term = copy.copy(ham)
    unit = copy.copy(p.iden)
    for i in np.arange(order, 0, -1, dtype=int):
        term /= i
        gate = unit + term
        term = np.einsum('ab,bw->aw', gate, ham)
    return gate

class gate(object):
    # input parameters
    # sites: involved sites
    # kind: tEvolP (plaquette) / tEvolV (vertex) / tEvolF (field) / Swap
    # para: parameter dictionary
    def __init__(self, sites, kind, para):
        # members of Gate
        self.sites = sites.copy()
        self.sites.sort()
        self.kind = kind

        expOrder = 100
        # create swap gate
        if ((self.kind == 'Swap') and (len(self.sites) == 2)):
            self.gate = np.zeros((2,2,2,2), dtype=complex)
            for a,b,c,d in product(range(2), repeat=4):
                if (a == d and b == c):
                    self.gate[a,b,c,d] = 1.0
        # first create local Hamiltonian
        # then convert to time evolution gate via toExpH
        elif ((self.kind == 'tEvolP') and (len(self.sites) == 4)):
            ham = np.einsum('ab,cd,ef,gh->abcdefgh', p.sx, p.sx, p.sx, p.sx) * (-para['p_g'])
            ham *= (-para['tau'] / 2) * 1.0j
            self.gate = toExpH4(ham, expOrder)
        elif ((self.kind == 'tEvolV') and (len(self.sites) == 4)):
            ham = np.einsum('ab,cd,ef,gh->abcdefgh', p.sz, p.sz, p.sz, p.sz) * (-para['v_U'])
            ham *= (-para['tau'] / 2) * 1.0j
            self.gate = toExpH4(ham, expOrder)
        elif ((self.kind == 'tEvolF') and (len(self.sites) == 1)):
            ham = p.sz * (-para['hz'])
            ham *= (-para['tau'] / 2) * 1.0j
            self.gate = toExpH1(ham, expOrder)
        else:
            print('Wrong parameter for gate construction.\n')
            sys.exit()

# clean redundant swap gates
def cleanGates(gateList):
    j = 0
    while j < len(gateList) - 1:
        if (gateList[j].kind == 'Swap' and gateList[j + 1].kind == 'Swap'
        and gateList[j].sites[0] == gateList[j + 1].sites[0]
        and gateList[j].sites[1] == gateList[j + 1].sites[1]):
            del gateList[j]
        else:
            j = j + 1


def makeGateList(sites, para):
    gateList = []

    # make field gates (no need for swap gates)
    if (para['hz'] != 0):
        for i in range(p.n):
            current_site = [i]
            gateList.append(gate(current_site, 'tEvolF', para))

    # make vertex gates
    if (para['v_U'] != 0):
        for i in np.arange(para['nx'] + 1, p.n - para['nx'] + 2, 2 * para['nx'] - 1, dtype=int):
            for j in np.arange(0, para['nx'] - 2, 1, dtype=int):
                u = i + j
                l = u + para['nx'] - 1
                r = l + 1
                d = l + para['nx']
                if (d > p.n):
                    d = d - p.n
                # move u to l - 1
                for site in np.arange(u, l - 1, 1, dtype=int):
                    gateList.append(gate([site, site + 1], 'Swap', para))
                # move d to r + 1
                for site in np.arange(d, r + 1, -1, dtype=int):
                    gateList.append(gate([site - 1, site], 'Swap', para))
                # evolution gate
                gateList.append(gate([l - 1, l, r, r + 1], 'tEvolV', para))
                # move l - 1 back to u
                for site in np.arange(l - 1, u, -1, dtype=int):
                    gateList.append(gate([site - 1, site], 'Swap', para))
                # move r + 1 back to d
                for site in np.arange(r + 1, d, 1, dtype=int):
                    gateList.append(gate([site, site + 1], 'Swap', para))

    # make plaquette gates
    if (para['p_g'] != 0):
        for i in np.arange(1, p.n, 2 * para['nx'] - 1, dtype=int):
            for j in np.arange(0, para['nx'] - 1, 1, dtype=int):
                u = i + j
                l = u + para['nx'] - 1
                r = l + 1
                d = l + para['nx']
                if (d > p.n):
                    d = d - p.n
                # move u to l - 1
                for site in np.arange(u, l - 1, 1, dtype=int):
                    gateList.append(gate([site, site + 1], 'Swap', para))
                # move d to r + 1
                for site in np.arange(d, r + 1, -1, dtype=int):
                    gateList.append(gate([site - 1, site], 'Swap', para))
                # evolution gate
                gateList.append(gate([l - 1, l, r, r + 1], 'tEvolP', para))
                # move l - 1 back to u
                for site in np.arange(l - 1, u, -1, dtype=int):
                    gateList.append(gate([site - 1, site], 'Swap', para))
                # move r + 1 back to d
                for site in np.arange(r + 1, d, 1, dtype=int):
                    gateList.append(gate([site, site + 1], 'Swap', para))

    # second order Trotter decomposition
    # b1.b2.b3....b3.b2.b1
    # append "reversed gateList" to "gateList"
    gateNum = len(gateList)
    for i in reversed(range(gateNum)):
        gateList.append(gateList[i])
    
    return gateList

def make_svd_posi(u,v):
    if (np.sum(u.flatten()) < 0 and np.sum(u.flatten()) < 0):
        u *= -1
        v *= -1

def svd_4site(tensor, cutoff):
    dim_l = tensor.shape[0]
    dim_r = tensor.shape[1]
    mat = np.einsum('imacegbdfh->iacbdegfhm', tensor)
    mat = np.reshape(mat, (dim_l * 16, dim_r * 16))

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
    m1 = np.einsum('iabs->isab', m1)
    m2 = np.dot(np.sqrt(mat_su), m2[0:retain_dim_u, :])
    m2 = np.reshape(m2, (retain_dim_u, 2, 2, retain_dim))
    m2 = np.einsum('scdn->sncd', m2)
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
    m3 = np.einsum('nefs->nsef', m3)
    m4 = np.dot(np.sqrt(mat_sv), m4[0:retain_dim_v, :])
    m4 = np.reshape(m4, (retain_dim_v, 2, 2, dim_r))
    m4 = np.einsum('sghm->smgh', m4)
    make_svd_posi(m3, m4)

    return m1, m2, m3, m4

def svd_2site(tensor, cutoff):
    dim_l = tensor.shape[0]
    dim_r = tensor.shape[1]
    mat = np.einsum('ijabcd->iabcdj', tensor)
    mat = np.reshape(mat, (dim_l * 4, dim_r * 4))

    u,s,v = np.linalg.svd(mat)
    s = s[np.where(np.abs(s[:]) > cutoff)[0]]
    retain_dim = s.shape[0]
    mat_s = np.diag(s)
    u = np.dot(u[:, 0:retain_dim], np.sqrt(mat_s))
    u = np.reshape(u, (dim_l, 2, 2, retain_dim))
    u = np.einsum('iabm->imab', u)
    v = np.dot(np.sqrt(mat_s), v[0:retain_dim, :])
    v = np.reshape(v, (retain_dim, 2, 2, dim_r))
    v = np.einsum('mcdj->mjcd', v)
    make_svd_posi(u, v)

    return u, v