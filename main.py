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
sites_on_str = [18,19,20,21]
for i in range(p.n):
    if i in sites_on_str:
        str_op[i][0,:,:,0] = p.sx
    else:
        str_op[i][0,:,:,0] = p.iden

# use adiabatic continuation to find the corresponding
# ground state |phi> when the magnetic field is on
new_op = copy.copy(str_op)
stepNum = int(p.para['ttotal'] / p.para['tau'])
for hx in np.linspace(0, -p.para['hx'], num=stepNum, dtype=float):
    # generate gates for one step of time evolution
    para['hx'] = hx
    gateList = gates.makeGateList(str_op, para)
    # phi = evol.gateTEvol(phi, gateList, para['tau'], para['tau'], cutoff, bondm)


print('Hello world')