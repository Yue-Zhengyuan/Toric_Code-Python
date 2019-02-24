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

# create local sz operator MPO (S)
# labelling: [site][L vir leg, R vir leg, U phys leg, D phys leg]
# len(MPO): number of sites
sz_op = []
for i in range(p.n):
    sz_op.append(np.zeros((1,2,2,1), dtype=complex))
    sz_op[i][0,:,:,0] = p.iden
coord = [2,2]
num = coord[1] * para['nx'] + coord[0]
sz_op[num][0,:,:,0] = p.sz

# create Ising ground state |psi_S>
# |+z>
zplus = []
for i in range(p.n):
    zplus.append(np.zeros((1,2,1), dtype=complex))
    zplus[i][0,0,0] = 1.0
# |-z>
zminus = []
for i in range(p.n):
    zminus.append(np.zeros((1,2,1), dtype=complex))
    zminus[i][0,1,0] = 1.0
zplus = np.asarray(zplus)
zminus = np.asarray(zminus)
# |psi_S> = (|+z> + |-z>)/sqrt(2)
psi_S = mps.sum(zplus, zminus, cutoff, bondm)
psi_S = mps.normalize(psi_S, 100, 100)
mps.printdata(psi_S)

# exp(+iHt) S exp(-iHt) |psi_S>
# adiabatic continuation of string operator
if mode == '0':
    stepNum = int(p.para['ttotal'] / p.para['tau'])
    # exp(-iHt) |psi_S>
    for hx in np.linspace(0, -p.para['hx'], num=stepNum, dtype=float):
        para['hx'] = hx
        gateList = ising_gates.makeGateList(sz_op, para)
        psi_S = mps.gateTEvol(psi_S, gateList, para['tau'], para['tau'], cutoff, bondm)
    # S exp(-iHt) |psi_S>
    psi_S = mps.applyMPOtoMPS(sz_op, psi_S, cutoff, bondm)
    # exp(+iHt) S exp(-iHt) |psi_S>
    for hx in reversed(np.linspace(0, -p.para['hx'], num=stepNum, dtype=float)):
        para['hx'] = hx
        gateList = ising_gates.makeGateList(sz_op, para)
        for g in gateList:
            g.gate = np.conj(g.gate)
        psi_S = mps.gateTEvol(psi_S, gateList, para['tau'], para['tau'], cutoff, bondm)

# quasi-adiabatic continuation of string operator
elif mode == '1':
    stepNum = 4
    # exp(-iHt) |psi_S>
    iter_list = np.linspace(0, -p.para['hx'], num=stepNum+1, dtype=float)
    iter_list = np.delete(iter_list, 0)
    for hx in iter_list:
        para['hx'] = hx
        gateList = ising_gates.makeGateList(sz_op, para)
        psi_S = mps.gateTEvol(psi_S, gateList, para['ttotal']/stepNum, para['tau'], cutoff, bondm)
    # S exp(-iHt) |psi_S>
    psi_S = mps.applyMPOtoMPS(sz_op, psi_S, cutoff, bondm)
    # exp(+iHt) S exp(-iHt) |psi_S>
    for hx in reversed(iter_list):
        para['hx'] = hx
        gateList = ising_gates.makeGateList(sz_op, para) # create exp(-iHt)
        for g in gateList:
            g.gate = np.conj(g.gate)                # convert to exp(+iHt)
        psi_S = mps.gateTEvol(psi_S, gateList, para['ttotal']/stepNum, para['tau'], cutoff, bondm)

# no-field Heisenberg evolution of string operator
elif mode == '2':
    # exp(-iHt) |psi_S>
    para['hx'] = 0.0
    gateList = ising_gates.makeGateList(sz_op, para)
    psi_S = mps.gateTEvol(psi_S, gateList, para['ttotal'], para['tau'], cutoff, bondm)
    psi_S = mps.normalize(psi_S, cutoff, bondm)
    mps.printdata(psi_S)
