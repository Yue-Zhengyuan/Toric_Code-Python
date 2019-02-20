# 
#   main_mps_ising.py
#   Toric_Code-Python
#   apply the gates and MPO to MPS (Ising Model)
#
#   created on Feb 18, 2019 by Yue Zhengyuan
#

import numpy as np
import ising_gates
import para_dict as p
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

# create directory to save result
# create folder to store results
# get system time
current_time = datetime.datetime.now()
result_dir = 'result_mps'
result_dir += '_' + str(current_time.year)
result_dir += '-' + str(current_time.month)
result_dir += '-' + str(current_time.day)
result_dir += '_' + str(current_time.hour)
result_dir += '_' + str(current_time.minute)
os.makedirs(result_dir, exist_ok=True)

# create local sz operator MPO (S)
# labelling: [site][L vir leg, R vir leg, U phys leg, D phys leg]
# len(MPO): number of sites
sz_op = []
for i in range(p.n):
    sz_op.append(np.zeros((1,2,2,1), dtype=complex))
coord = [5,5]
num = coord[1] * para['nx'] + coord[0]
sz_op[num][0,:,:,0] = p.sz

# create Ising ground state |psi_S>
# |+z>
zplus = []
for i in range(10):
    zplus.append(np.zeros((1,2,1), dtype=complex))
    zplus[i][0,0,0] = 1.0
# |-z>
zminus = []
for i in range(10):
    zminus.append(np.zeros((1,2,1), dtype=complex))
    zminus[i][0,1,0] = 1.0
zplus = np.asarray(zplus)
zminus = np.asarray(zminus)
# |psi_S> = (|+z> + |-z>)/sqrt(2)
psi_S = mps.sum(zplus, zminus, cutoff, bondm)
psi_S = mps.normalize(psi_S, 100, 100)

# save parameter of current running
with open(result_dir + '/parameters.txt', 'w+') as file:
    file.write(json.dumps(p.para))  # use json.loads to do the reverse
    file.write('\nSite of Sz\n')
    file.write(str(coord))    # coordinate
    file.write(str(num))      # site number

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
    mps.save_to_file(psi_S, result_dir + '/adiab_psi.txt')

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
    mps.save_to_file(psi_S, result_dir + '/quasi_psi.txt')

# no-field Heisenberg evolution of string operator
elif mode == '2':
    # exp(-iHt) |psi_S>
    para['hx'] = 0.0
    gateList = ising_gates.makeGateList(sz_op, para)
    psi_S = mps.gateTEvol(psi_S, gateList, para['ttotal'], para['tau'], cutoff, bondm)
    # S exp(-iHt) |psi_S>
    psi_S = mps.applyMPOtoMPS(sz_op, psi_S, cutoff, bondm)
    # exp(+iHt) S exp(-iHt) |psi_S>
    for g in gateList:
        g.gate = np.conj(g.gate)
    psi_S = mps.gateTEvol(psi_S, gateList, para['ttotal'], para['tau'], cutoff, bondm)
    mps.save_to_file(psi_S, result_dir + '/no_field_psi.txt')
