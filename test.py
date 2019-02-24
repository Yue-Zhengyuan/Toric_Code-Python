# 
#   main_mps_ising.py
#   Toric_Code-Python
#   apply the gates and MPO to MPS (Ising Model)
#
#   created on Feb 18, 2019 by Yue Zhengyuan
#

import numpy as np
import ising_gates
import ising_para_dict as p
import lattice as lat
import mps
import mpo
import sys
import copy
import time
import datetime
import os
import json

cutoff = 100
bondm = 32
para = copy.copy(p.para)
# clear magnetic field
para['hx'] = 0
para['hy'] = 0
para['hz'] = 0
# mode = sys.argv[1]
mode = '2'

# create local sx operator MPO
# labelling: [site][L vir leg, R vir leg, U phys leg, D phys leg]
# len(MPO): number of sites
sx_op = []
for i in range(p.n):
    sx_op.append(np.zeros((1,2,2,1), dtype=complex))
    sx_op[i][0,:,:,0] = p.iden
coord = [2,2]
num = coord[1] * para['nx'] + coord[0]
sx_op[num][0,:,:,0] = p.sx

# |+z>
psi_up = []
for i in range(p.n):
    psi_up.append(np.zeros((1,2,1), dtype=complex))
    psi_up[i][0,0,0] = 1.0

result = copy.copy(psi_up)

start = time.perf_counter()

# exp(+iHt) S exp(-iHt) |psi_S>
# adiabatic continuation of string operator
if mode == '0':
    ttotal = para['ttotal']
    tstep = para['tau']
    stepNum = int(ttotal/tstep + (1e-9 * (ttotal/tstep)))
    # exp(-iHt) |psi_S>
    for hx in np.linspace(0, p.para['hx'], num=stepNum, dtype=float):
        para['hx'] = hx
        gateList = ising_gates.makeGateList(sx_op, para)
        result = mps.gateTEvol(result, gateList, para['tau'], para['tau'], cutoff, bondm)
    # S exp(-iHt) |psi_S>
    result = mps.applyMPOtoMPS(sx_op, result, cutoff, bondm)
    # exp(+iHt) S exp(-iHt) |psi_S>
    para['J'] *= -1
    for hx in reversed(np.linspace(0, p.para['hx'], num=stepNum, dtype=float)):
        para['hx'] = -hx
        gateList = ising_gates.makeGateList(sx_op, para)
        result = mps.gateTEvol(result, gateList, para['tau'], para['tau'], cutoff, bondm)
    mps.printdata(result)

# no-field Heisenberg evolution of string operator
elif mode == '2':
    # test one step of time evolution
    # exp(-iHt) |psi_S>
    para['hx'] = 1.0
    gateList = ising_gates.makeGateList(sx_op, para)
    result = mps.gateTEvol(result, gateList, para['tau'], para['tau'], cutoff, bondm)
    mps.printdata(result)

finish = time.perf_counter()
elapsed = finish - start
print("Elapsed time:", elapsed, "s\n")