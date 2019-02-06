# 
#   main.py
#   Toric_Code-Python
#   apply the gates and MPO to MPS
#
#   created on Jan 21, 2019 by Yue Zhengyuan
#

import numpy as np
import gates
import para_dict as p
import gateTEvol as evol
import sys
import copy
import time

def applyMPOtoMPS(mpo, mps, cutoff):
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
    for site in range(siteNum):
        group = np.einsum('ijab,kbl->ikajl',mpo[site],mps[site])
        if site == 0:
            result = group
        else:
            result = np.tensordot(result, group, axes=([0,1],[-2,-1]))
    # restore MPO form by SVD and truncate virtual links
    # set orthogonality center at the middle of the MPO
    evol.position(result, int(siteNum/2), cutoff)
    return result

def overlap(mps1, mps2, cutoff):
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
    for site in range(siteNum):
        group = np.einsum('iaj,kal->ikjl',np.conj(mps1[site]),mps2[site])
        if site == 0:
            result = group
        else:
            result = np.tensordot(result, group, axes=([0,1],[-2,-1]))
    # restore MPO form by SVD and truncate virtual links
    # set orthogonality center at the middle of the MPO
    evol.position(result, int(siteNum/2), cutoff)
    return result

# create string operator MPO
# labelling: [site][L vir leg, R vir leg, U phys leg, D phys leg]
# len(MPO): number of sites
str_op = []
for i in range(p.n):
    str_op.append(np.zeros((1,2,2,1), dtype=complex))
sites_on_str = [18,19,20,21]
for i in range(p.n):
    if i in sites_on_str:
        str_op[i][0,:,:,0] = p.sx
    else:
        str_op[i][0,:,:,0] = p.iden

# create lattice MPS (spin 1/2)
# labelling: [site][L vir leg, R vir leg, phys leg]
mps = []
for i in range(p.n):
    str_op.append(np.zeros((1,1,1), dtype=complex))

# generate gates for one step of time evolution
t_start = time.time()
gateList = gates.makeGateList(str_op, p.para)
t_end = time.time()
print('Gate generation time: ', t_end-t_start, ' s')
# apply gates to the string operator MPO
t_start = time.time()
evol.gateTEvol(mps, gateList, 1.0, p.para['tau'])
t_end = time.time()
print('Gate evolution time: ', t_end-t_start, ' s')
print('Hello world')