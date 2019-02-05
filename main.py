# 
#   main.py
#   Toric_Code-Python
#   apply the gates to string operator
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

# multiply two MPO's
def multMPO(mpo1, mpo2, cutoff):
    siteNum = len(mpo1)
    if (len(mpo2) != siteNum):
        print('Number of sites of the two MPOs do not match.')
        sys.exit()
    # dimension check
    for site in range(siteNum):
        if mpo1[site].shape[2] != mpo2[site].shape[1]:
            print('Physical leg dimensions of the two MPOs do not match.')
            sys.exit()
    # get shape for the result MPO
    newshape = []
    for i in range(siteNum):
        newshape.append(mpo2[i].shape[1])
    for i in range(siteNum):
        newshape.append(mpo1[i].shape[2])
    newshape.append(mpo2[siteNum-1].shape[-1] * mpo1[siteNum-1].shape[-1])
    newshape.insert(0, mpo2[siteNum-1].shape[0] * mpo1[siteNum-1].shape[0])
    # contraction
    for site in range(siteNum):
        group = np.einsum('kcal,ijab->kicblj',mpo2[site],mpo1[site])
        if site == 0:
            result = group
        else:
            oldshape = result.shape
            result = np.einsum('kicblj,ljfemn->kicfbemn', result, group)
            result = np.reshape(result,
            (oldshape[0], oldshape[1], oldshape[2]*group.shape[2], oldshape[3]*group.shape[3], group.shape[4], group.shape[5]))
    # restore MPO form
    virDim = [result.shape[0]*result.shape[1], result.shape[4]*result.shape[5]]
    result = np.reshape(result, (virDim[0], result.shape[2], result.shape[3], virDim[1]))
    result = np.reshape(result, newshape)
    # rearrange indices
    for i in range(siteNum):
        for j in np.arange(siteNum+1, i+1, -1, dtype=int):
            result = np.swapaxes(result, j-1, j)
    # do svd to truncate virtual links
    # and set orthogonality center at the middle of the MPO
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

# generate gates for one step of time evolution
t_start = time.time()
gateList = gates.makeGateList(str_op, p.para)
t_end = time.time()
print('Gate generation time: ', t_end-t_start, ' s')
# apply gates to the string operator MPO
t_start = time.time()
evol.gateTEvol(str_op, gateList, 1.0, p.para['tau'])
t_end = time.time()
print('Gate evolution time: ', t_end-t_start, ' s')
print('Hello world')