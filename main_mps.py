# 
#   main_mps.py
#   Toric_Code-Python
#   apply the gates and MPO to MPS
#
#   created on Feb 18, 2019 by Yue Zhengyuan
#

import numpy as np
import gates
import para_dict as p
import lattice as lat
import mps
# import mpo
import sys
import copy
import time
import datetime
import os
import json

cutoff = 100
bondm = 32
args = copy.copy(p.args)
# clear magnetic field
args['hx'] = 0
args['hy'] = 0
args['hz'] = 0
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
    bond_list.append(lat.lat(bond[0:2],bond[2],args['nx']))
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

# save parameter of current running
with open(result_dir + '/parameters.txt', 'w+') as file:
    file.write(json.dumps(p.args))  # use json.loads to do the reverse
    file.write('\nSites on String\n')
    file.write(str(bond_on_str))    # coordinate
    file.write(str(bond_list))      # bond number

# exp(+iHt) S exp(-iHt) |psi>
# adiabatic continuation of string operator
if mode == '0':
    ttotal = args['ttotal']
    tstep = args['tau']
    stepNum = int(ttotal/tstep + (1e-9 * (ttotal/tstep)))
    # exp(-iHt) |psi_S>
    for hx in np.linspace(0, p.args['hx'], num=stepNum, dtype=float):
        args['hx'] = hx
        gateList = gates.makeGateList(str_op, args)
        result = mps.gateTEvol(result, gateList, args['tau'], args['tau'], args=args)
    # S exp(-iHt) |psi_S>
    result = mps.applyMPOtoMPS(str_op, result, args=args)
    # exp(+iHt) S exp(-iHt) |psi_S>
    args['J'] *= -1
    for hx in reversed(np.linspace(0, p.args['hx'], num=stepNum, dtype=float)):
        args['hx'] = -hx
        gateList = gates.makeGateList(str_op, args)
        result = mps.gateTEvol(result, gateList, args['tau'], args['tau'], args=args)
    mps.save_to_file(result, result_dir + '/adiab_psi.txt')

elif mode == '1':
    ttotal = args['ttotal']
    tstep = args['tau']
    stepNum = 5
    # exp(-iHt) |psi_S>
    for hx in np.linspace(0, p.args['hx'], num=stepNum, dtype=float):
        args['hx'] = hx
        gateList = gates.makeGateList(str_op, args)
        result = mps.gateTEvol(result, gateList, args['tau'], args['tau'], args=args)
    # S exp(-iHt) |psi_S>
    result = mps.applyMPOtoMPS(str_op, result, args=args)
    # exp(+iHt) S exp(-iHt) |psi_S>
    args['J'] *= -1
    for hx in reversed(np.linspace(0, p.args['hx'], num=stepNum, dtype=float)):
        args['hx'] = -hx
        gateList = gates.makeGateList(str_op, args)
        result = mps.gateTEvol(result, gateList, args['tau'], args['tau'], args=args)
    mps.save_to_file(result, result_dir + '/quasi_psi.txt')

# no-field Heisenberg evolution of string operator
elif mode == '2':
    # exp(-iHt) |psi_S>
    args['hx'] = 0.0
    gateList = gates.makeGateList(str_op, args)
    result = mps.gateTEvol(result, gateList, args['ttotal'], args['tau'], args=args)
    # S exp(-iHt) |psi_S>
    result = mps.applyMPOtoMPS(str_op, result, args=args)
    # exp(+iHt) S exp(-iHt) |psi_S>
    for g in gateList:
        g.gate = np.conj(g.gate)
    result = mps.gateTEvol(result, gateList, args['ttotal'], args['tau'], args=args)
    mps.save_to_file(result, result_dir + '/no_field_psi.txt')
