# 
#   main_mps.py
#   Toric_Code-Python
#   apply the gates and MPO to MPS
#
#   created on Feb 18, 2019 by Yue Zhengyuan
#

import numpy as np
import old_gates
import gates
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
para['hx'] = 0.0
para['hy'] = 0.0
para['hz'] = 0.0
# mode = sys.argv[1]
mode = '2'

# create string operator MPO (S)
# labelling: [site][L vir leg, R vir leg, U phys leg, D phys leg]
# len(MPO): number of sites
str_op = []
for i in range(p.n):
    str_op.append(np.zeros((1,2,2,1), dtype=complex))
# string along x
bond_on_str = [(8,10,'r'),(9,10,'r'),(10,10,'r'),(11,10,'r'),(12,10,'r')]
# string along y
# bond_on_str = [(10,8,'d'),(10,9,'d'),(10,10,'d'),(10,11,'d'),(10,12,'d')]
# convert coordinate to unique number
bond_list = []
for bond in bond_on_str:
    bond_list.append(lat.lat(bond[0:2],bond[2],para['nx']))
for i in range(p.n):
    if i in bond_list:
        str_op[i][0,:,:,0] = p.sx
    else:
        str_op[i][0,:,:,0] = p.iden

# create Ising ground state |psi>
psi = []
for i in range(p.n):
    psi.append(np.zeros((1,2,1), dtype=complex))
    psi[i][0,0,0] = 1.0

# exp(+iHt) S exp(-iHt) |psi>
# quasi-adiabatic evolution of string operator
if mode == '1':
    stepNum = 4
    # exp(-iHt) |psi>
    iter_list = np.linspace(0, -p.para['hx'], num=stepNum+1, dtype=float)
    iter_list = np.delete(iter_list, 0)
    for hx in iter_list:
        para['hx'] = hx
        gateList = old_gates.makeGateList(str_op, para)
        psi = mps.gateTEvol(psi, gateList, para['ttotal']/stepNum, para['tau'], cutoff, bondm)
    # S exp(-iHt) |psi>
    psi = mps.applyMPOtoMPS(str_op, psi, cutoff, bondm)
    # exp(+iHt) S exp(-iHt) |psi>
    for hx in iter_list:
        para['hx'] = hx
        gateList = old_gates.makeGateList(str_op, para)
        for g in gateList:
            g.gate = np.conj(g.gate)
        psi = mps.gateTEvol(psi, gateList, para['tau']/stepNum, para['tau'], cutoff, bondm)

if mode == '2':
    psi1 = copy.copy(psi)
    psi2 = copy.copy(psi)
    gateList = old_gates.makeGateList(str_op, para)
    psi1 = mps.gateTEvol(psi1, gateList, para['ttotal'], para['tau'], cutoff, bondm)
    gateList = gates.makeGateList(str_op, para)
    psi2 = mps.gateTEvol(psi2, gateList, para['ttotal'], para['tau'], cutoff, bondm)
    mps.save_to_file(psi1, "separate_field.txt")
    mps.save_to_file(psi2, "together_field.txt")

print('Hello world!')