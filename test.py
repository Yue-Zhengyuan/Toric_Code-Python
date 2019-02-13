# 
#   test.py
#   Toric_Code-Python
#   apply the gates and MPO to MPS
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

cutoff = 1000
bondm = 32
para = p.para

# create string operator MPO (S)
# labelling: [site][L vir leg, R vir leg, U phys leg, D phys leg]
# len(MPO): number of sites
str_op = []
for i in range(p.n):
    str_op.append(np.zeros((1,2,2,1), dtype=complex))
sites_on_str = [60,61,71,90,109,128,138,139]
for i in range(p.n):
    if i in sites_on_str:
        str_op[i][0,:,:,0] = p.sx
    else:
        str_op[i][0,:,:,0] = p.iden

# no-field Heisenberg evolution of string operator
adiab_op = copy.copy(str_op)
para['hx'] = 0.0
gateList = gates.makeGateList(str_op, para)
print('finished gate creation')
adiab_op = mpo.gateTEvol(adiab_op, gateList, para['ttotal'], para['tau'], cutoff, bondm)
mpo.save_to_file(adiab_op, 'no_field_op.txt')
