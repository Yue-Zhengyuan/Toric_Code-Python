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
import gnd_state
import sys
import copy
import time
import datetime
import os
import json

args = copy.copy(p.args)
# clear magnetic field
args['hx'] = 0.0
args['hy'] = 0.0
args['hz'] = 0.0

# create closed string operator MPO enclosing different area(S)
bond_on_str = []
# listing edges of the closed string
# area = 1
bond_on_str.append(((2,2,'r'),(3,2,'d'),(2,3,'r'),(2,2,'d'),1))
bond_on_str.append(((1,1,'r'),(2,1,'d'),(1,2,'r'),(1,1,'d'),1))
# area = 2
bond_on_str.append(((1,1,'r'),(2,1,'d'),(2,2,'d'),(1,3,'r'),(1,2,'d'),(1,1,'d'),2))
bond_on_str.append(((1,2,'r'),(2,2,'r'),(3,2,'d'),(2,3,'r'),(1,3,'r'),(1,2,'d'),2))
# area = 3
bond_on_str.append(((1,1,'r'),(2,1,'d'),(2,2,'r'),(3,2,'d'),
(2,3,'r'),(1,3,'r'),(1,2,'d'),(1,1,'d'),3))
bond_on_str.append(((1,1,'r'),(2,1,'r'),(3,1,'r'),(4,1,'d'),
(3,2,'r'),(2,2,'r'),(1,2,'r'),(1,1,'d'),3))
# area = 4
bond_on_str.append(((1,1,'r'),(2,1,'r'),(3,1,'d'),(3,2,'d'),
(2,3,'r'),(1,3,'r'),(1,2,'d'),(1,1,'d'),4))
bond_on_str.append(((1,1,'r'),(2,1,'r'),(3,1,'d'),(3,2,'r'),
(4,2,'d'),(3,3,'r'),(2,3,'r'),(2,2,'d'),(1,2,'r'),(1,1,'d'),4))
bond_on_str.append(((1,1,'r'),(2,1,'r'),(3,1,'r'),(4,1,'d'),
(3,2,'r'),(2,2,'r'),(2,2,'d'),(1,3,'r'),(1,2,'d'),(1,1,'d'),4))
bond_on_str.append(((0,2,'r'),(1,2,'r'),(2,2,'r'),(3,2,'r'),
(4,2,'d'),(3,3,'r'),(2,3,'r'),(1,3,'r'),(0,3,'r'),(0,2,'d'),4))
# area = 5
bond_on_str.append(((1,1,'r'),(2,1,'r'),(3,1,'d'),(3,2,'r'),
(4,2,'d'),(3,3,'r'),(2,3,'r'),(1,3,'r'),(1,2,'d'),(1,1,'d'),5))
bond_on_str.append(((0,1,'r'),(1,1,'r'),(2,1,'d'),(2,2,'r'),
(3,2,'r'),(4,2,'d'),(3,3,'r'),(2,3,'r'),(1,3,'r'),(1,2,'d'),
(0,2,'r'),(0,1,'d'),5))
bond_on_str.append(((0,1,'r'),(1,1,'r'),(2,1,'r'),(3,1,'r'),
(4,1,'d'),(4,2,'d'),(3,3,'r'),(3,2,'d'),(2,2,'r'),(1,2,'r'),
(0,2,'r'),(0,1,'d'),5))
# area = 6
bond_on_str.append(((1,1,'r'),(2,1,'r'),(3,1,'r'),(4,1,'d'),
(4,2,'d'),(3,3,'r'),(2,3,'r'),(1,3,'r'),(1,2,'d'),(1,1,'d'),6))
bond_on_str.append(((0,1,'r'),(1,1,'r'),(2,1,'r'),(3,1,'d'),
(3,2,'r'),(4,2,'d'),(3,3,'r'),(2,3,'r'),(1,3,'r'),(1,2,'d'),
(0,2,'r'),(0,1,'d'),6))

# create result directory
# get system time
nowtime = datetime.datetime.now()
result_dir = '_'.join(['result_mps', str(nowtime.year), str(nowtime.month), 
str(nowtime.day), str(nowtime.hour), str(nowtime.minute)])
os.makedirs(result_dir, exist_ok=True)

# save parameters
with open(result_dir + '/parameters.txt', 'w+') as file:
    file.write(json.dumps(p.args))  # use json.loads to do the reverse
    file.write('\n\nUsing String Operators:')
    for string in bond_on_str:
        file.write('\n\n')
        file.write(str(string))    # coordinate

# create Toric Code ground state |psi>
psi = gnd_state.gnd_state_builder(args)

tstart = time.perf_counter()

# adiabatic evolution: exp(-iHt)|psi> (With field along z)
stepNum = int(p.args['ttotal']/p.args['tau'])
iterlist = np.linspace(0, p.args['hz'], num = stepNum+1, dtype=float)
iterlist = np.delete(iterlist, 0)
for hz in iterlist:
    args['hz'] = hz
    gateList = gates.makeGateList(psi, args)
    psi = mps.gateTEvol(psi, gateList, args['ttotal']/stepNum, args['tau'], args=args)
tend = time.perf_counter()
with open(result_dir + '/parameters.txt', 'a+') as file:
    print("Adiabatic evolution:", tend-tstart, "s\n")

# quasi adiabatic evolution:
# exp(-iH't) exp(-iHt) |psi>
stepNum = 4
iterlist = np.linspace(0, p.args['hz'], num = stepNum+1, dtype=float)
iterlist = np.delete(iterlist, 0)
for hz in iterlist:
    args['hz'] = hz
    gateList = gates.makeGateList(psi, args)
    psi = mps.gateTEvol(psi, gateList, args['ttotal']/stepNum, args['tau'], args=args)
tend = time.perf_counter()
with open(result_dir + '/parameters.txt', 'a+') as file:
    file.write("Quasi-adiabatic evolution:" + str(tend-tstart) + "s\n")

# apply dressed string
# <psi| exp(+iHt) exp(+iH't) S exp(-iH't) exp(-iHt) |psi>
for string in bond_on_str:
    str_op = []
    for i in range(p.n):
        str_op.append(np.zeros((1,2,2,1), dtype=complex))
    # convert coordinate to unique number in 1D
    bond_list = []
    for bond in string[0:-1]:
        bond_list.append(lat.lat(bond[0:2],bond[2],args['nx']))
    for i in range(p.n):
        if i in bond_list:
            str_op[i][0,:,:,0] = p.sx
        else:
            str_op[i][0,:,:,0] = p.iden
    result = mps.matElem(psi, str_op, psi)
    with open(result_dir + '/result.txt', 'a+') as file:
        file.write(str(string[-1]) + "\t" + str(result) + '\n')