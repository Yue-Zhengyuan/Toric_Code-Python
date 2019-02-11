# 
#   test.py
#   Toric_Code-Python
#   apply the gates and MPO to MPS (test)
#
#   created on Jan 21, 2019 by Yue Zhengyuan
#

import numpy as np
import gates
import para_dict as p
import mps
import mpo
import sys
import copy
import time

cutoff = 100
bondm = 32

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
print('Number of gates: ', len(gateList))

# apply gates to the MPS to get new |phi> = exp(-iHt)|psi>
t_start = time.time()
str_op = mpo.gateTEvol(str_op, gateList, 0.01, p.para['tau'], cutoff, bondm)
t_end = time.time()
print('Gate evolution time: ', t_end-t_start, ' s')

# write resulting MPS |phi> into txt file
mpo.save_to_file(str_op, 'string_operator.txt')
print('Hello world')