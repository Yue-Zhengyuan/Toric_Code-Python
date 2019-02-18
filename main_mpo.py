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
mode = sys.argv[1]
# mode = '2'

# create directory to save result
# create folder to store results
# get system time
current_time = datetime.datetime.now()
result_dir = 'result'
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

# bond_on_str = [(2,3,'r'),(3,3,'r'),(4,3,'d'),(4,4,'d'),(4,5,'d'),(4,6,'r'),(5,6,'r')]
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

# save parameter of current running
with open(result_dir + '/parameters.txt', 'w+') as file:
    file.write(json.dumps(p.para)) # use `json.loads` to do the reverse
    file.write('\nSites on String\n')
    file.write(str(bond_on_str))

# adiabatic continuation of string operator
if mode == '0':
    adiab_op = copy.copy(str_op)
    stepNum = int(p.para['ttotal'] / p.para['tau'])
    for hz in np.linspace(0, -p.para['hx'], num=stepNum, dtype=float):
        para['hz'] = hz
        gateList = gates.makeGateList(str_op, para)
        adiab_op = mpo.gateTEvol(adiab_op, gateList, para['tau'], para['tau'], cutoff, bondm)
    mpo.save_to_file(adiab_op, result_dir + '/adiab_op.txt')

# quasi-adiabatic continuation of string operator
elif mode == '1':
    quasi_op = copy.copy(str_op)
    stepNum = 4
    iter_list = np.linspace(0, -p.para['hx'], num=stepNum+1, dtype=float)
    iter_list = np.delete(iter_list, 0)
    for hz in iter_list:
        para['hz'] = hz
        gateList = gates.makeGateList(str_op, para)
        quasi_op = mpo.gateTEvol(quasi_op, gateList, para['ttotal']/stepNum, para['tau'], cutoff, bondm)
    mpo.save_to_file(quasi_op, result_dir + '/quasi_op.txt')

# no-field Heisenberg evolution of string operator
elif mode == '2':
    no_field_op = copy.copy(str_op)
    para['hz'] = 0.0
    gateList = gates.makeGateList(str_op, para)
    no_field_op = mpo.gateTEvol(no_field_op, gateList, para['ttotal'], para['tau'], cutoff, bondm)
    mpo.save_to_file(no_field_op, result_dir + '/no_field_op.txt')
