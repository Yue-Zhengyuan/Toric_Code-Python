# 
#   test.py
#   Toric_Code-Python
#   turning off vertex U for MPO evolution
#
#   created on Jan 21, 2019 by Yue Zhengyuan
#

import numpy as np
import gates
import para_dict as p
import lattice as lat
import gnd_state
import mps
import mpo
import sys
import copy
import time
import datetime
import os
import json

args = copy.copy(p.args)
# turn off vertex
args['U'] = 0
args['scale'] = True
# clear magnetic field
args['hx'] = 0
args['hy'] = 0
args['hz'] = 0

# create directory to save result
# create folder to store results
# get system time
current_time = datetime.datetime.now()
result_dir = 'result_mpotest'
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
bond_on_str = ((1,1,'r'),(2,1,'r'),(3,1,'d'),(3,2,'d'),
(2,3,'r'),(1,3,'r'),(1,2,'d'),(1,1,'d'))\
# convert coordinate to unique number
bond_list = []
for bond in bond_on_str:
    bond_list.append(lat.lat(bond[0:2],bond[2],args['nx']))

for i in range(p.n):
    if i in bond_list:
        str_op[i][0,:,:,0] = p.sx
    else:
        str_op[i][0,:,:,0] = p.iden

# save parameter of current running
with open(result_dir + '/parameters.txt', 'w+') as file:
    file.write(json.dumps(p.args)) # use `json.loads` to do the reverse
    file.write('\nSites on String\n')
    file.write(str(bond_on_str))

# adiabatic continuation of string operator
tstart = time.perf_counter()

adiab_op = copy.copy(str_op)
stepNum = int(p.args['ttotal'] / p.args['tau'])
args['scale'] = True
for hz in np.linspace(0, -p.args['hx'], num=stepNum, dtype=float):
    args['hz'] = hz
    gateList = gates.makeGateList(str_op, args)
    adiab_op = mpo.gateTEvol(adiab_op, gateList, args['tau'], args['tau'], args=args)
mpo.save_to_file(adiab_op, result_dir + '/adiab_op.txt')
tend = time.perf_counter()
with open(result_dir + '/parameters.txt', 'a+') as file:
    file.write('\nTime evolution: ' + str(tend-tstart) + 's\n')

# create Ising ground state to visualize spreading of string operator
psi_up = []
for i in range(p.n):
    psi_up.append(np.zeros((1,2,1), dtype=complex))
    psi_up[i][0,0,0] = 1.0

# create Toric Code ground state
psi = gnd_state.gnd_state_builder(args)

args['scale'] = False
result = mps.applyMPOtoMPS(adiab_op, psi_up, args=args)
mpo.save_to_file(result, result_dir + '/string_on_ising.txt')

result = mps.applyMPOtoMPS(str_op, psi, args=args)
result = mps.overlap(psi, result)
mpo.save_to_file(result, result_dir + '/string_on_toric.txt')
