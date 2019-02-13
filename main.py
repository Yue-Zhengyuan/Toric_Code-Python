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
import mps
import mpo
import sys
import copy
import time

cutoff = 100
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

# adiabatic continuation of string operator
adiab_op = copy.copy(str_op)
stepNum = int(p.para['ttotal'] / p.para['tau'])
for hx in np.linspace(0, -p.para['hx'], num=stepNum, dtype=float):
    para['hx'] = hx
    gateList = gates.makeGateList(str_op, para)
    adiab_op = mpo.gateTEvol(adiab_op, gateList, para['tau'], para['tau'], cutoff, bondm)
mpo.save_to_file(adiab_op, 'adiab_op.txt')

# quasi-adiabatic continuation of string operator
quasi_op = copy.copy(str_op)
stepNum = 5
iter_list = np.linspace(0, -p.para['hx'], num=stepNum+1, dtype=float)
iter_list = np.delete(iter_list, 0)
for hx in iter_list:
    para['hx'] = hx
    gateList = gates.makeGateList(str_op, para)
    quasi_op = mpo.gateTEvol(quasi_op, gateList, para['ttotal']/stepNum, para['tau'], cutoff, bondm)
mpo.save_to_file(quasi_op, 'quasi_op.txt')

print('Hello world')