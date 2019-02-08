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
import mps
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

# create lattice MPS (spin 1/2) |psi>
# labelling: [site][L vir leg, R vir leg, phys leg]
psi = []
for i in range(p.n):
    psi.append(np.zeros((1,2,1), dtype=complex))
    # with all spin up
    psi[i][0,0,0] = 1.0
    # with all spin down
    # psi[i][0,1,0] = 1.0

# generate gates for one step of time evolution
t_start = time.time()
gateList = gates.makeGateList(str_op, p.para)
t_end = time.time()
print('Gate generation time: ', t_end-t_start, ' s')
print('Number of gates: ', len(gateList))

# apply gates to the MPS to get new |phi> = exp(-iHt)|psi>
t_start = time.time()
phi = evol.gateTEvol(psi, gateList, 1, p.para['tau'], cutoff, bondm)
t_end = time.time()
print('Gate evolution time: ', t_end-t_start, ' s')
result = mps.overlap(phi, phi, cutoff, bondm)

# apply string operator to MPS to get |xi> = S|phi>
xi = mps.applyMPOtoMPS(str_op, phi, cutoff, bondm)

# calculate <phi|xi> = <psi|exp(+iHt) S exp(-iHt)|psi>
result = mps.overlap(phi, xi, cutoff, bondm)

print('Hello world')